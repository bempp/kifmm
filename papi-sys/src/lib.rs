#![allow(non_upper_case_globals)] // allow top-level consts if any

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::all
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    include!(concat!(env!("OUT_DIR"), "/codegen.rs"));
}

#[allow(unused)]
pub use bindings::*;

#[allow(unused)]
#[cfg(test)]
mod tests {

    use lazy_static::lazy_static;

    lazy_static! {
        static ref IS_PAPI_INITED: bool = {
            do_papi_init();
            true
        };
    }

    use super::*;

    fn do_papi_init() {
        unsafe {
            let ver = PAPI_library_init(PAPI_VER_CURRENT);
            assert_eq!(ver, PAPI_VER_CURRENT);
        }

        let is_inited = unsafe { PAPI_is_initialized() };
        assert_ne!(is_inited, PAPI_NOT_INITED as i32);
    }

    #[test]
    fn get_real_cyc() {
        let cycles = unsafe { PAPI_get_real_cyc() };
        assert!(cycles >= 0);
    }

    // #[test]
    // fn get_num_counters() {
    //     let num_hwcntrs = unsafe { PAPI_num_counters() };
    //     assert!(num_hwcntrs >= 0);
    // }
}
