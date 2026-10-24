use std::process::ExitCode;
use std::time::Instant;

use solver::profiles::{Profile, profile_id};
use solver::table::{Table, Transitions};

fn main() -> ExitCode {
    let start = Instant::now();
    let t = Transitions::build();
    let mut table = match Table::new(t.len()) {
        Ok(table) => table,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match table.sweep(&t, 0) {
        Ok(solved) => println!(
            "Solved {solved} classes in {:.1} s.",
            start.elapsed().as_secs_f64()
        ),
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    }
    let anchor = profile_id(&Profile::FailureFatal { st: 240 }) as usize;
    println!("V(240, 0, 240, 0) = {:.16}", table.get(anchor, anchor));
    println!("V(0, 0, 0, 0)     = {:.16}", table.get(0, 0));
    ExitCode::SUCCESS
}
