use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValueError {
    #[error("Probability mass value {0} invalid")]
    InvalidProbability(f64),

    #[error("Value {0} invalid")]
    InvalidValue(f64),

    #[error("Squandered time {0} invalid")]
    InvalidSt(usize),

    #[error("You supplied lag {0}; use 1 through 60.")]
    InvalidLag(usize),

    #[error("Total time dead {0} invalid")]
    InvalidTtd(usize),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Profile {
    Eligible { st: usize, ttd: usize },
    FailureFatal { st: usize },
}
/// Calculates the win probability from a game value at a node in the game tree
pub fn win_probability(v: f64) -> Result<f64, ValueError> {
    if !(-1.0 <= v && v <= 1.0) {
        return Err(ValueError::InvalidValue(v));
    } else {
        Ok((v + 1.0) / 2.0)
    }
}

pub fn eligible(s: usize, t: usize) -> bool {
    s < 240 && s + t <= 240
}

/// Frozen probability model assumption for the game solver
pub fn revival_probability(s: usize, t: usize) -> f64 {
    if !eligible(s, t) {
        0.0
    } else {
        0.95 * (1.0 - (s as f64) / 240.0) * (0.75_f64.powf((t as f64) / 60.0))
    }
}

/// Canonical representation of one profile (st, ttd)
pub fn canonical(s: usize, t: usize) -> Result<Profile, ValueError> {
    if s > 299 {
        return Err(ValueError::InvalidSt(s));
    }
    if t > 300 || (1..=59).contains(&t) {
        return Err(ValueError::InvalidTtd(t));
    }
    Ok(if eligible(s, t) {
        Profile::Eligible { st: s, ttd: t }
    } else {
        Profile::FailureFatal { st: s }
    })
}

/// Success child given the current checker finds the handkerchief in lag seconds
pub fn success_child(profile: &Profile, lag: usize) -> Result<Option<Profile>, ValueError> {
    if !(1 <= lag && lag <= 60) {
        return Err(ValueError::InvalidLag(lag));
    }
    match profile {
        Profile::FailureFatal { st } => {
            let new_s: usize = st + lag;
            if new_s >= 300 {
                Ok(None)
            } else {
                Ok(Some(Profile::FailureFatal { st: new_s }))
            }
        }
        Profile::Eligible { st, ttd } => {
            let new_s: usize = st + lag;
            if new_s >= 300 {
                Ok(None)
            } else {
                Ok(Some(canonical(new_s, *ttd)?))
            }
        }
    }
}

/// Revived child given the current checker survives being injected
pub fn revived_child(profile: &Profile) -> Result<Option<Profile>, ValueError> {
    match profile {
        Profile::FailureFatal { .. } => Ok(None),
        Profile::Eligible { st, ttd } => Ok(Some(canonical(0, st + ttd + 60)?)),
    }
}

/// Calculates the catalog ID of a profile in closed form.
/// Eligible IDs ascend by TTD, then ST. TTD zero holds 240 profiles and each
/// TTD t in 60..=240 holds 241 - t, so the block offset for t >= 60 is
/// 240 + (423 - t)(t - 60) / 2. Failure-fatal IDs follow at 16_711 + st.
pub fn profile_id(profile: &Profile) -> u32 {
    match *profile {
        Profile::Eligible { st, ttd: 0 } => st as u32,
        Profile::Eligible { st, ttd } => (240 + (423 - ttd) * (ttd - 60) / 2 + st) as u32,
        Profile::FailureFatal { st } => (ELIGIBLE_COUNT + st) as u32,
    }
}

pub const ELIGIBLE_COUNT: usize = 16_711;
pub const PROFILE_COUNT: usize = 17_011;

pub fn potential(profile: &Profile) -> usize {
    match profile {
        Profile::Eligible { st, ttd } => st + ttd,
        Profile::FailureFatal { st } => st + 301,
    }
}

pub fn profile_catalog() -> Result<Vec<Profile>, ValueError> {
    let mut profiles = Vec::with_capacity(PROFILE_COUNT);
    for t in std::iter::once(0).chain(60..=300) {
        for s in 0..300 {
            if eligible(s, t) {
                profiles.push(canonical(s, t)?);
            }
        }
    }
    for s in 0..300 {
        profiles.push(Profile::FailureFatal { st: s })
    }
    return Ok(profiles);
}
