#[rustfmt::skip] // Ordered, not Alphabetized
use log::Level::{Error, Warn, Info, Debug, Trace};
use crate::IntoLevelIter;

macro_rules! test_iteration {
    ($(
        [$($exp_item:expr),*] : {
            $($into_iter:expr),+ $(,)?
        }
    )+) => { {$(
        let expected = vec![$($exp_item),*];
        $(
            println!("Testing iteration: '{}' == {:?}", stringify!($into_iter), expected);
            let mut iter = ($into_iter).into_level_iter();
            for &expected_item in &expected {
                assert_eq!(Some(expected_item), iter.next());
            }
            assert_eq!(None, iter.next());
        )*
    )*} }
}

#[test]
fn test_into_level_iter() {
    test_iteration!(
        [Error, Warn, Info, Debug, Trace] : {
            ..,
            Error..,
            Error..=Trace,
            [Error, Warn, Info, Debug, Trace],
        }
        [Error, Warn, Info, Debug] : {
            Error..Trace,
            Error..=Debug,
            [Error, Warn, Info, Debug],
        }
        [Warn, Info, Debug] : {
            Warn..Trace,
            Warn..=Debug,
            [Warn, Info, Debug],
        }
        [Debug, Trace] : {
            Debug..,
            Debug..=Trace,
            [Debug, Trace],
        }
        [Error] : {
            Error..Warn,
            Error..=Error,
            Error,
            [Error],
        }
        [Trace] : {
            Trace..,
            Trace..=Trace,
            Trace,
            [Trace],
        }
    )
}

// #[test]
// // This test requires unwinding panic
// fn test_invalid_ranges() {
//     std::panic::catch_unwind(|| (..Error).into_level_iter()).unwrap_err();
//     std::panic::catch_unwind(|| (Error..Error).into_level_iter()).unwrap_err();
//     std::panic::catch_unwind(|| (Warn..Error).into_level_iter()).unwrap_err();
//     std::panic::catch_unwind(|| (Warn..=Error).into_level_iter()).unwrap_err();
// }

#[test]
#[should_panic(expected = "Level range was backward or empty")]
fn test_invalid_level_range_1() {
    (..Error).into_level_iter();
}
#[test]
#[should_panic(expected = "Level range was backward or empty")]
fn test_invalid_level_range_2() {
    (Error..Error).into_level_iter();
}
#[test]
#[should_panic(expected = "Level range was backward or empty")]
fn test_invalid_level_range_3() {
    (Warn..Error).into_level_iter();
}
#[test]
#[should_panic(expected = "Level range was backward or empty")]
fn test_invalid_level_range_4() {
    (Warn..=Error).into_level_iter();
}
