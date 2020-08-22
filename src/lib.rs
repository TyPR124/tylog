#![deny(missing_docs)]
//! This crate provides a simple logging interface, intended for applications that have
//! relatively simple logging needs and particularly for applications that can define
//! their logging needs at compile time. It is built on top of the [`log`] crate.
//!
//! It is not recommended to use the [`log`] crate directly in conjuntion with this crate.
//! Instead, this crate re-exports `log`'s items.
//!
//! # Example
//!
//! ```rust
//! // Build and initialize some (arbitrary for the sake of example) logger.
//! use tylog::Level::{Error, Info, Debug, Trace};
//! tylog::new_logger()
//!     .with_format("{level} {msg}")
//!     // Can enable just a specific level for a target.
//!     .enable_stdout(Info)
//!     // Can enable a range of levels for a target.
//!     .enable_stderr(Error..Info) // Note: exclusive range exludes Info
//!     .with_format("{time} {level} {fileLineOrFile} - {msg}")
//!     .enable_file(Error..=Info, "./foo.log")?
//!     .enable_file(Debug.., "./debug_trace.log")?
//!     .enable_file(.., "./everything.log")?
//!     // Can enable a specific set of levels, excluding all others.
//!     .enable_file(&[Error, Debug], "./error_debug.log")?
//!     .init()?;
//! // Do some (arbitrary for the sake of example) logging.
//! use tylog::{error, warn, info, debug, trace};
//! error!("This is an error message.");
//! warn!("This is a warning message.");
//! info!("This is an informational.");
//! debug!("This is a debug message.");
//! trace!("This is a trace message.");
//! // May be a good idea to flush the logs before app ends.
//! tylog::flush();
//! # //std::fs::remove_file("./foo.log").unwrap_or_else(drop);
//! # //std::fs::remove_file("./debug_trace.log").unwrap_or_else(drop);
//! # //std::fs::remove_file("./error_debug.log").unwrap_or_else(drop);
//! # //std::fs::remove_file("./everything.log").unwrap_or_else(drop);
//! # Ok::<(), Box<dyn std::error::Error + 'static>>(())
//! ```
//!
//! [`log`]: https://docs.rs/log

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    fmt::Write as _,
    fs::File,
    io::Write as _,
    ops::{Bound, RangeBounds},
    path::{Path, PathBuf},
};

#[rustfmt::skip] // Ordered, not Alphabetized
use log::Level::{Error, Warn, Info, Debug, Trace};

pub use log::{self, debug, error, info, trace, warn, Level, LevelFilter};

const LEVEL_COUNT: usize = 5;
/// Gets optimized to no-op
#[inline(always)]
fn lvl_to_usize(lvl: log::Level) -> usize {
    match lvl {
        Error => 1,
        Warn => 2,
        Info => 3,
        Debug => 4,
        Trace => 5,
    }
}
/// Probably can be optimized to no-op in many cases, but not really tested.
#[inline(always)]
fn lvl_from_usize(lvl: usize) -> log::Level {
    match lvl {
        1 => Error,
        2 => Warn,
        3 => Info,
        4 => Debug,
        5 => Trace,
        _ => unreachable!(),
    }
}

/// The default log format. See [`LogBuilder::with_format`] for more information.
///
/// [`LogBuilder::with_format`]: struct.LogBuilder.html#method.with_format
pub const DEFAULT_LOG_FORMAT: &str = "{time} {level} {fileLine} {msg}";
/// The default time format. See [`LogBuilder::with_format`] and [the time crate] and for more information.
///
/// [`LogBuilder::with_format`]: struct.LogBuilder.html#method.with_format
/// [the time crate]: https://time-rs.github.io/time/time/index.html#formatting
pub const DEFAULT_TIME_FORMAT: &str = "%0Y-%0m-%0d %0H:%0M:%0S %0z";

/// The default level strings. `"[ERROR]"`, `"[WARN] "`, `"[INFO] "`, `"[DEBUG]"`, and `"[TRACE]"`.
pub const DEFAULT_LEVEL_STRINGS: LevelStrings = LevelStrings {
    strings: ["", "[ERROR]", "[WARN] ", "[INFO] ", "[DEBUG]", "[TRACE]"],
};

/// A convenience trait for types which can be passed as a list of [`log::Level`]s.
///
/// This trait is implemented for [`log::Level`], [`RangeBounds`]`<`[`log::Level`]`>`, and [`[`][array][`log::Level`][`log::Level`][`; 1]`][array] through [`[`][array][`log::Level`][`log::Level`][`; 5]`][array].
///
/// [`log::Level`]: log/enum.Level.html
/// [`RangeBounds`]: https://doc.rust-lang.org/stable/std/ops/trait.RangeBounds.html
/// [array]: https://doc.rust-lang.org/stable/std/primitive.array.html
pub trait IntoLevelIter<I: Iterator<Item = log::Level>>: Sized {
    /// Consume self and produce the iterator.
    fn into_level_iter(self) -> I;
}

#[doc(hidden)]
/// A range of [`log::Level`]s.
///
/// [`log::Level`]: log/enum.Level.html
#[derive(Debug)]
pub struct LevelRange {
    range: std::ops::RangeInclusive<usize>,
}

impl Iterator for LevelRange {
    type Item = log::Level;
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(lvl_from_usize)
    }
}

impl<R: RangeBounds<log::Level>> IntoLevelIter<LevelRange> for R {
    fn into_level_iter(self) -> LevelRange {
        let first = match self.start_bound() {
            Bound::Excluded(_) => unimplemented!("No such thing as an excluded start bound"),
            Bound::Included(&level) => lvl_to_usize(level),
            Bound::Unbounded => lvl_to_usize(Error),
        };
        let last = match self.end_bound() {
            Bound::Excluded(&level) => lvl_to_usize(level) - 1,
            Bound::Included(&level) => lvl_to_usize(level),
            Bound::Unbounded => lvl_to_usize(Trace),
        };
        assert!(
            first <= last,
            "Level range was backward or empty. Note: Error < Warn < Info < Debug < Trace"
        );
        LevelRange {
            range: first..=last,
        }
    }
}

impl IntoLevelIter<std::iter::Once<log::Level>> for log::Level {
    fn into_level_iter(self) -> std::iter::Once<log::Level> {
        std::iter::once(self)
    }
}

type CopiedFromSliceIter<'a> = std::iter::Copied<std::slice::Iter<'a, log::Level>>;

impl<'a> IntoLevelIter<CopiedFromSliceIter<'a>> for &'a [log::Level; 1] {
    fn into_level_iter(self) -> CopiedFromSliceIter<'a> {
        self.iter().copied()
    }
}
impl<'a> IntoLevelIter<CopiedFromSliceIter<'a>> for &'a [log::Level; 2] {
    fn into_level_iter(self) -> CopiedFromSliceIter<'a> {
        self.iter().copied()
    }
}
impl<'a> IntoLevelIter<CopiedFromSliceIter<'a>> for &'a [log::Level; 3] {
    fn into_level_iter(self) -> CopiedFromSliceIter<'a> {
        self.iter().copied()
    }
}
impl<'a> IntoLevelIter<CopiedFromSliceIter<'a>> for &'a [log::Level; 4] {
    fn into_level_iter(self) -> CopiedFromSliceIter<'a> {
        self.iter().copied()
    }
}
impl<'a> IntoLevelIter<CopiedFromSliceIter<'a>> for &'a [log::Level; 5] {
    fn into_level_iter(self) -> CopiedFromSliceIter<'a> {
        self.iter().copied()
    }
}

#[derive(Default)]
struct EnabledLevels {
    // One extra slot to avoid subtraction in indexing
    enabled: [bool; LEVEL_COUNT + 1],
}
impl EnabledLevels {
    pub fn from<I: Iterator<Item = log::Level>>(levels: impl IntoLevelIter<I>) -> Self {
        let mut enabled = Self::default();
        for level in levels.into_level_iter() {
            enabled[level] = true;
        }
        enabled
    }
}
impl std::ops::Index<log::Level> for EnabledLevels {
    type Output = bool;
    fn index(&self, level: log::Level) -> &bool {
        &self.enabled[lvl_to_usize(level)]
    }
}
impl std::ops::IndexMut<log::Level> for EnabledLevels {
    fn index_mut(&mut self, level: log::Level) -> &mut bool {
        &mut self.enabled[lvl_to_usize(level)]
    }
}
impl EnabledLevels {
    pub fn merge(&mut self, other: &Self) {
        for (vs, vo) in self.enabled.iter_mut().zip(other.enabled.iter()).skip(1) {
            *vs |= vo;
        }
    }
    pub fn max(&self) -> log::LevelFilter {
        if self[Trace] {
            log::LevelFilter::Trace
        } else if self[Debug] {
            log::LevelFilter::Debug
        } else if self[Info] {
            log::LevelFilter::Info
        } else if self[Warn] {
            log::LevelFilter::Warn
        } else if self[Error] {
            log::LevelFilter::Error
        } else {
            log::LevelFilter::Off
        }
    }
}

/// Trait for a type which can be written through a shared reference.
///
/// This is automatically implented for all `&T where T: SharedWrite`.
/// Additionally, this is implemented on `File`, `Mutex<T: Write>`, and `RwLock<T: Write>`.
trait SharedWrite: Send + Sync {
    /// Write all bytes.
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()>;
    /// Flush the writer.
    fn flush(&self) -> std::io::Result<()>;
}

impl SharedWrite for File {
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()> {
        std::io::Write::write_all(&mut &*self, buf)
    }
    fn flush(&self) -> std::io::Result<()> {
        std::io::Write::flush(&mut &*self)
    }
}
struct Stdout;
impl SharedWrite for Stdout {
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()> {
        std::io::stdout().write_all(buf)
    }
    fn flush(&self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}

struct Stderr;
impl SharedWrite for Stderr {
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()> {
        std::io::stderr().write_all(buf)
    }
    fn flush(&self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

impl<W: std::io::Write + Send> SharedWrite for std::sync::Mutex<W> {
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()> {
        self.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .write_all(buf)
    }
    fn flush(&self) -> std::io::Result<()> {
        self.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .flush()
    }
}

impl<W: std::io::Write + Send + Sync> SharedWrite for std::sync::RwLock<W> {
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()> {
        std::sync::RwLock::write(self)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .write_all(buf)
    }
    fn flush(&self) -> std::io::Result<()> {
        std::sync::RwLock::write(self)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .flush()
    }
}

impl<T: SharedWrite> SharedWrite for &T {
    fn write_all(&self, buf: &[u8]) -> std::io::Result<()> {
        (*self).write_all(buf)
    }
    fn flush(&self) -> std::io::Result<()> {
        (*self).flush()
    }
}

struct LogTarget {
    writer: &'static dyn SharedWrite,
    fmt: &'static [FmtPart],
    /// This ends up being a double-deref. I'm doing this becuase
    /// we need to hash this field for each target. Hashing one &(Sized)
    /// is much better than [&(!Sized); 6]. Note, this has not been
    /// performance tested.
    lvl_strings: &'static LevelStrings,
    enabled: EnabledLevels,
}

#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct LevelStrings {
    /// One extra slot to avoid subtraction in indexing
    strings: [&'static str; LEVEL_COUNT + 1],
}
impl std::cmp::PartialEq for LevelStrings {
    fn eq(&self, other: &Self) -> bool {
        self.strings[1..] == other.strings[1..]
    }
}

impl From<[&'static str; 5]> for LevelStrings {
    fn from(strings: [&'static str; 5]) -> Self {
        let strings = [
            "", strings[0], strings[1], strings[2], strings[3], strings[4],
        ];
        Self { strings }
    }
}
impl<F: Fn(log::Level) -> &'static str> From<F> for LevelStrings {
    fn from(f: F) -> Self {
        Self {
            strings: ["", f(Error), f(Warn), f(Info), f(Debug), f(Trace)],
        }
    }
}
// Note - think about with_level_strings before impl any new From<T> for LevelStrings

impl std::ops::Index<log::Level> for LevelStrings {
    type Output = str;
    fn index(&self, level: log::Level) -> &str {
        self.strings[lvl_to_usize(level)]
    }
}

/// An intermediate type constructed via [`new_logger`].
///
/// [`new_logger`]: fn.new_logger.html
pub struct LogBuilder {
    current_fmt: &'static [FmtPart],
    current_lvl_strings: &'static LevelStrings,
    // Memoize formats. Important for later memoizing formatted log messages.
    known_fmts: HashMap<String, &'static [FmtPart]>,
    // Memoize level strings. Important for later memoizing formatted log messages.
    known_lvl_strings: Vec<&'static LevelStrings>,
    file_targets: HashMap<PathBuf, LogTarget>,
    other_targets: Vec<LogTarget>,
}

impl Default for LogBuilder {
    fn default() -> Self {
        let mut known_fmts = HashMap::new();
        let current_fmt = build_static_format_parts(DEFAULT_LOG_FORMAT);
        known_fmts.insert(DEFAULT_LOG_FORMAT.to_owned(), current_fmt);
        let current_lvl_strings = &DEFAULT_LEVEL_STRINGS;
        let known_lvl_strings = vec![current_lvl_strings];
        LogBuilder {
            current_fmt,
            current_lvl_strings,
            known_fmts,
            known_lvl_strings,
            file_targets: HashMap::new(),
            other_targets: vec![],
        }
    }
}

/// Constructs a [`LogBuilder`].
///
/// [`LogBuilder`]: struct.LogBuilder.html
#[must_use]
pub fn new_logger() -> LogBuilder {
    LogBuilder::default()
}

impl LogBuilder {
    /// Sets the formatting used by yet-to-be-enabled logs. Any logs enabled prior to this call will use the prior formatting, or [`DEFAULT_LOG_FORMAT`]
    /// if this method was never called.
    ///
    /// The format string is comprised of a series of literals and tags in any order. For example, the default format, `"{time} {level} {fileLine} {msg}"`
    /// consists of four tags, `{time}`, `{level}`, `{fileLine}`, and `{msg}`, and three literals which are the spaces in between. Tags are case-insensative.
    ///
    /// Tags may have an argument. An argument is added by appending `:` followed by the argument literal. For example, `"{foo:bar}"`, here the tag is "foo" and
    /// the argument is "bar". The only tag which uses an argument is the `{time}` tag. The argument of the `{time}` is not required, in which case it defaults
    /// to [`DEFAULT_TIME_FORMAT`]. The time format is specified by [the time crate].
    ///
    /// The list of all tags currently supported is
    /// * `{Time}` - the time the entry was logged. Takes an optional [format](https://time-rs.github.io/time/time/index.html#formatting) argument.
    /// * `{Level}` - the level of the log. See [`with_level_strings`] for more info.
    /// * `{Msg}` - the log message.
    /// * `{File}` - the file which triggered the log. This may be absent.
    /// * `{Line}` - the line which triggered the log. This may be absent.
    /// * `{FileLine}` - the file and line, if both are present. Formatted as `relative/path/to/foo.rs:42`.
    /// * `{FileLineOrFile}` - the same as `{FileLine}` if both are present, otherwise the same as `{File}`.
    ///
    /// [`DEFAULT_LOG_FORMAT`]: constant.DEFAULT_LOG_FORMAT.html
    /// [`DEFAULT_TIME_FORMAT`]: constant.DEFAULT_TIME_FORMAT.html
    /// [the time crate]: https://time-rs.github.io/time/time/index.html#formatting
    /// [`with_level_strings`]: struct.LogBuilder.html#method.with_level_strings
    #[must_use = "Note: the .init() method must be called to actually enable logging"]
    pub fn with_format(mut self, format: &str) -> Self {
        self.current_fmt = if let Some(fmt) = self.known_fmts.get(format) {
            fmt
        } else {
            let fmt = build_static_format_parts(format);
            self.known_fmts.insert(format.to_owned(), fmt);
            fmt
        };
        self
    }
    /// Use the given level strings for yet-to-be-enabled logs. Any logs enabled prior to this call will use the prior level strings, or [`DEFAULT_LEVEL_STRINGS`]
    /// if this method was never called.
    ///
    /// The argument to this method can be either an array of `[&'static str; 5]` or a `Fn(tylog::Level) -> &'static str` which will be called exactly once per level.
    /// In the case of an array, the values should be ordered from "Error" to "Trace".
    ///
    /// The default level strings are `["[ERROR]", "[WARN] ", "[INFO] ", "[DEBUG]", "[TRACE]"]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// tylog::new_logger()
    ///     .with_level_strings(|lvl| match lvl {
    ///         Error => "Error",
    ///         Warn => "Warning",
    ///         Info => "Info",
    ///         Debug => "Debug",
    ///         Trace => "Trace",
    ///     })
    ///     .init()?;
    /// # Ok::<(), Box<dyn std::error::Error + 'static>>(())
    /// ```
    #[must_use = "Note: the .init() method must be called to actually enable logging"]
    pub fn with_level_strings(mut self, into_lvl_strings: impl Into<LevelStrings>) -> Self {
        let new_lvl_strings = into_lvl_strings.into();
        self.current_lvl_strings = if let Some(lvl_strings) = self
            .known_lvl_strings
            .iter()
            .find(|&&&ls| ls == new_lvl_strings)
        {
            lvl_strings
        } else {
            let lvl_strings = Box::leak(Box::new(new_lvl_strings));
            self.known_lvl_strings.push(lvl_strings);
            lvl_strings
        };
        self
    }
    /// Enable logging to stdout. Example:
    ///
    /// ```rust
    /// use tylog::Level::Info;
    /// tylog::new_logger()
    ///     // Log informational and higher to stdout.
    ///     .enable_stdout(Info..)
    ///     .init()?;
    /// # Ok::<(), Box<dyn std::error::Error + 'static>>(())
    /// ```
    #[must_use = "Note: the .init() method must be called to actually enable logging"]
    pub fn enable_stdout<I: Iterator<Item = log::Level>>(
        mut self,
        levels: impl IntoLevelIter<I>,
    ) -> Self {
        self.other_targets.push(LogTarget {
            writer: &Stdout,
            lvl_strings: self.current_lvl_strings,
            fmt: self.current_fmt,
            enabled: EnabledLevels::from(levels),
        });
        self
    }
    /// Enable logging to stderr. Example:
    ///
    /// ```rust
    /// use tylog::Level::{Error, Info};
    /// tylog::new_logger()
    ///     // Log error up to but excluding informational to stderr.
    ///     .enable_stderr(Error..Info)
    ///     .init()?;
    /// # Ok::<(), Box<dyn std::error::Error + 'static>>(())
    /// ```
    #[must_use = "Note: the .init() method must be called to actually enable logging"]
    pub fn enable_stderr<I: Iterator<Item = log::Level>>(
        mut self,
        levels: impl IntoLevelIter<I>,
    ) -> Self {
        self.other_targets.push(LogTarget {
            writer: &Stderr,
            lvl_strings: self.current_lvl_strings,
            fmt: self.current_fmt,
            enabled: EnabledLevels::from(levels),
        });
        self
    }
    /// Enable logging to the file passed in. Example:
    ///
    /// ```rust
    /// use tylog::Level::{Error, Info, Debug, Trace};
    /// tylog::new_logger()
    ///     // Log error up to and including warning to foo.log
    ///     .enable_file(Error..=Info, "./foo.log").expect("Failed to open foo.log")
    ///     // Log just trace messages to trace.log
    ///     .enable_file(Trace, "./trace.log")?
    ///     .init()?;
    /// # std::fs::remove_file("./foo.log").unwrap_or_else(drop);
    /// # std::fs::remove_file("./trace.log").unwrap_or_else(drop);
    /// # Ok::<(), Box<dyn std::error::Error + 'static>>(())
    /// ```
    #[must_use = "Note: the .init() method must be called to actually enable logging"]
    pub fn enable_file<I: Iterator<Item = log::Level>>(
        mut self,
        levels: impl IntoLevelIter<I>,
        path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let full_path = canonicalize_file_path(path.as_ref())?;
        if self.file_targets.contains_key(&full_path) {
            // Intentially using (non-canonicolized) path in error message
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "File {} is already opened by tylog",
                    path.as_ref().display()
                ),
            ));
        }
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&full_path)?;
        if self
            .file_targets
            .insert(
                full_path,
                LogTarget {
                    writer: Box::leak(Box::new(file)),
                    fmt: self.current_fmt,
                    lvl_strings: self.current_lvl_strings,
                    enabled: EnabledLevels::from(levels),
                },
            )
            .is_some()
        {
            panic!("HashMap was mutated elsewhere while holding exclusive borrow")
        }
        Ok(self)
    }
    /// Initialize the logger. There can only be one logger, and this will fail if one was already set.
    pub fn init(self) -> Result<(), log::SetLoggerError> {
        let mut logs = self.other_targets;
        logs.reserve(self.file_targets.len());
        logs.extend(self.file_targets.into_iter().map(|(_, v)| v));
        let mut all_enabled = EnabledLevels::default();
        for LogTarget { enabled, .. } in &logs {
            all_enabled.merge(enabled);
        }
        let max_enabled = all_enabled.max();
        let logger = Logger {
            enabled: all_enabled,
            logs,
        };
        log::set_logger(Box::leak(Box::new(logger)))?;
        log::set_max_level(max_enabled);
        Ok(())
    }
}

struct Logger {
    enabled: EnabledLevels,
    logs: Vec<LogTarget>,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum FmtPart {
    Time { time_fmt: &'static str },
    Literal { lit: &'static str },
    Level,
    Message,
    File,
    Line,
    FileLine,
    FileLineOrFile,
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.enabled[metadata.level()]
    }
    fn log(&self, record: &log::Record) {
        let mut memoizer = MemoizingLogFormatter::new(record);
        for target in &self.logs {
            if target.enabled[record.level()] {
                let formatted = memoizer.format(target.fmt, target.lvl_strings);
                target
                    .writer
                    .write_all(formatted.as_bytes())
                    .unwrap_or_else(drop);
            }
        }
    }
    fn flush(&self) {
        for log in &self.logs {
            log.writer.flush().unwrap_or_else(drop);
        }
    }
}
struct MemoizingLogFormatter<'r> {
    record: &'r log::Record<'r>,
    // If the args (msg) are to be formatted, just do it once.
    args: Option<String>,
    // If the time is to be acquired, just do it once.
    time: Option<time::OffsetDateTime>,
    // If the time is to be formatted, just do it once per format.
    time_fmts: HashMap<(usize, usize), String, fxhash::FxBuildHasher>,
    // If the record is to be formatted, just do it once per format.
    fmts: HashMap<(usize, usize, usize), String, fxhash::FxBuildHasher>,
}
impl<'r> MemoizingLogFormatter<'r> {
    pub fn new(record: &'r log::Record) -> Self {
        Self {
            record,
            args: None,
            time: None,
            time_fmts: HashMap::default(),
            fmts: HashMap::default(),
        }
    }
    /// Returns the format of the record with the given fmt parts and level strings, and stores the result in case it is needed again.
    pub fn format(
        &mut self,
        parts: &'static [FmtPart],
        level_strings: &'static LevelStrings,
    ) -> &str {
        let key = (
            parts.as_ptr() as usize,
            parts.len(),
            level_strings as *const LevelStrings as usize,
        );
        let time = &mut self.time;
        let args = &mut self.args;
        let time_fmts = &mut self.time_fmts;
        let record = self.record;
        self.fmts.entry(key).or_insert_with(|| {
            let mut out = String::new();
            for part in parts {
                match part {
                    FmtPart::Literal { lit } => out += lit,
                    FmtPart::Time { time_fmt } => {
                        let time_key = (time_fmt.as_ptr() as usize, time_fmt.len());
                        let time_str = &*time_fmts.entry(time_key).or_insert_with(|| {
                            let dt = match time {
                                Some(dt) => *dt,
                                None => {
                                    let dt = time::OffsetDateTime::now_local();
                                    *time = Some(dt);
                                    dt
                                }
                            };
                            dt.format(time_fmt)
                        });
                        out += time_str
                    }
                    FmtPart::Level => out += &level_strings[record.level()],
                    FmtPart::Message => {
                        let args = match args {
                            Some(args) => &*args,
                            None => {
                                *args = Some(format!("{}", record.args()));
                                args.as_ref().unwrap()
                            }
                        };
                        out += args
                    }
                    FmtPart::File => {
                        if let Some(file_name) = record.file() {
                            out += file_name
                        }
                    }
                    FmtPart::Line => {
                        if let Some(line) = record.line() {
                            write!(out, "{}", line).unwrap_or_else(drop)
                        }
                    }
                    FmtPart::FileLine => {
                        if let (Some(file), Some(line)) = (record.file(), record.line()) {
                            write!(out, "{}:{}", file, line).unwrap_or_else(drop)
                        }
                    }
                    FmtPart::FileLineOrFile => match (record.file(), record.line()) {
                        (Some(file), Some(line)) => {
                            write!(out, "{}:{}", file, line).unwrap_or_else(drop)
                        }
                        (Some(file), None) => out += file,
                        _ => {}
                    },
                }
            }
            out += "\n";
            out
        })
    }
}

fn build_static_format_parts(format_str: &str) -> &'static [FmtPart] {
    use nom::bytes::complete::{escaped_transform, is_not, tag};
    use nom::character::complete::one_of;
    use nom::multi::{fold_many_m_n, many1};
    use nom::sequence::delimited;
    use nom::IResult;

    fn literal(input: &str) -> IResult<&str, FmtPart> {
        let (remaining, lit_str) = escaped_transform(is_not("\\{"), '\\', one_of("\\{}"))(input)?;
        let lit = Box::leak(lit_str.into_boxed_str());
        Ok((remaining, FmtPart::Literal { lit }))
    }
    fn tag_and_args(input: &str) -> IResult<&str, (&str, Option<&str>)> {
        let (remaining, tag_str) = is_not(":")(input)?;
        let (remaining, ()) = fold_many_m_n(0, 1, tag(":"), (), |(), _| ())(remaining)?;
        let tag_args = if remaining.is_empty() {
            None
        } else {
            Some(remaining)
        };
        Ok(("", (tag_str, tag_args)))
    }
    fn non_literal(input: &str) -> IResult<&str, FmtPart> {
        let (remaining, part_tag) = delimited(tag("{"), is_not("}"), tag("}"))(input)?;
        let (tag_remaining, (tag_ident, tag_args)) = tag_and_args(part_tag)?;
        assert!(tag_remaining.is_empty());

        let part = if tag_ident.eq_ignore_ascii_case("level") {
            FmtPart::Level
        } else if tag_ident.eq_ignore_ascii_case("msg") {
            FmtPart::Message
        } else if tag_ident.eq_ignore_ascii_case("time") {
            FmtPart::Time {
                time_fmt: tag_args
                    .map(str::to_owned)
                    .map(String::into_boxed_str)
                    .map(|boxed| &*Box::leak(boxed))
                    .unwrap_or(DEFAULT_TIME_FORMAT),
            }
        } else if tag_ident.eq_ignore_ascii_case("file") {
            FmtPart::File
        } else if tag_ident.eq_ignore_ascii_case("line") {
            FmtPart::Line
        } else if tag_ident.eq_ignore_ascii_case("FileLine") {
            FmtPart::FileLine
        } else if tag_ident.eq_ignore_ascii_case("FileLineOrFile") {
            FmtPart::FileLineOrFile
        } else {
            panic!("Unrecognized tag '{}'", tag_ident);
        };

        Ok((remaining, part))
    }
    fn part(input: &str) -> IResult<&str, FmtPart> {
        if input.is_empty() {
            let err = nom::error::make_error(input, nom::error::ErrorKind::NonEmpty);
            // I think there's an error
            Err(nom::Err::Error(err))
        } else if input.starts_with('{') {
            non_literal(input)
        } else {
            literal(input)
        }
    }

    let (remaining, v) = many1(part)(format_str).unwrap();
    let v: Vec<FmtPart> = v;
    assert!(remaining.is_empty());

    Box::leak(v.into_boxed_slice())
}

#[test]
fn test_build_fmt_parts() {
    use FmtPart::*;
    assert_eq!(
        dbg!(build_static_format_parts(
            "Some\\\\thing \\{happened! {level} Whatever shall we do?"
        )),
        &[
            Literal {
                lit: "Some\\thing {happened! "
            },
            Level,
            Literal {
                lit: " Whatever shall we do?"
            }
        ]
    );
    assert_eq!(
        dbg!(build_static_format_parts(
            r#"Some\\thing \{happened!\} {level} Whatever shall we do?"#
        )),
        &[
            Literal {
                lit: "Some\\thing {happened!} "
            },
            Level,
            Literal {
                lit: " Whatever shall we do?"
            }
        ]
    );
}

/// Flushes all logs, globally, ensuring any pending data is written out. If a logger was never initialized, this does nothing.
pub fn flush() {
    log::logger().flush()
}

fn canonicalize_file_path(path: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let path = path.as_ref();
    match std::fs::canonicalize(path) {
        Ok(canonicalized) => Ok(canonicalized),
        Err(err) if matches!(err.kind(), std::io::ErrorKind::NotFound) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            drop(file);
            std::fs::canonicalize(path)
        }
        err => err,
    }
}
