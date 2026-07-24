use std::os::unix::net::UnixStream;

use redface_core::{prelude::*, Config, DaemonRequest, DaemonResponse};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let config = Config::load_default()?;
	let mut conn = UnixStream::connect(&config.socket)?;
	let uid = unsafe { libc::geteuid() };
	let req = DaemonRequest::Authenticate {
		client: "check".into(),
		user: uid.to_string(),
		timeout: Some(-1),
		show_osd: true,
	};
	req.write_to(&mut conn)?;
	let res = DaemonResponse::read_from(&mut conn)?;
	match res {
		DaemonResponse::AuthSuccess => println!("Result success"),
		DaemonResponse::AuthError(msg) => println!("Result {msg}"),
	}
	Ok(())
}
