use std::collections::BTreeMap;
use std::ffi::CStr;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::path::Path;

use pam::constants::{PAM_ERROR_MSG, PAM_TEXT_INFO, PamFlag, PamResultCode};
use pam::conv::Conv;
use pam::module::{PamHandle, PamHooks};
use redface_runtime::{Config, Status, read_res, write_auth_req};
use users::os::unix::UserExt;

struct RedfacePam;
pam::pam_hooks!(RedfacePam);

impl PamHooks for RedfacePam {
	fn sm_authenticate(pamh: &mut PamHandle, args: Vec<&CStr>, _flags: PamFlag) -> PamResultCode {
		authenticate(pamh, args)
	}

	fn sm_setcred(_pamh: &mut PamHandle, _args: Vec<&CStr>, _flags: PamFlag) -> PamResultCode {
		PamResultCode::PAM_SUCCESS
	}
}

fn authenticate(pamh: &mut PamHandle, args: Vec<&CStr>) -> PamResultCode {
	let config = match Config::load_default() {
		Ok(config) => config,
		Err(_) => return PamResultCode::PAM_AUTHINFO_UNAVAIL,
	};

	let sock_stat = match fs::metadata(&config.socket) {
		Ok(stat) => stat,
		Err(err) if err.kind() == std::io::ErrorKind::NotFound => return PamResultCode::PAM_IGNORE,
		Err(_) => return PamResultCode::PAM_AUTH_ERR,
	};

	if sock_stat.uid() != 0 {
		return PamResultCode::PAM_AUTH_ERR;
	}

	let user_name = match pamh.get_user(None) {
		Ok(user) => user,
		Err(err) => return err,
	};
	let user = match users::get_user_by_name(&user_name) {
		Some(user) => user,
		None => return PamResultCode::PAM_USER_UNKNOWN,
	};

	let arg_map = parse_args(args);
	if let Some(template) = arg_map.get("ifexist") {
		let conditional_file = render_user_template(template, &user);
		if !Path::new(&conditional_file).exists() {
			return PamResultCode::PAM_IGNORE;
		}
	}

	let mut conn = match UnixStream::connect(&config.socket) {
		Ok(conn) => conn,
		Err(_) => return PamResultCode::PAM_CRED_UNAVAIL,
	};

	let _ = send_message(pamh, "Scanning face...", false);

	let client = arg_map
		.get("client")
		.map(String::as_str)
		.filter(|value| !value.is_empty())
		.unwrap_or("pam");

	if write_auth_req(&mut conn, &user.uid().to_string(), client).is_err() {
		let _ = send_message(pamh, "Daemon error", true);
		return PamResultCode::PAM_CRED_UNAVAIL;
	}

	let res = match read_res(&mut conn) {
		Ok(res) => res,
		Err(_) => {
			let _ = send_message(pamh, "Daemon error", true);
			return PamResultCode::PAM_CRED_UNAVAIL;
		}
	};

	if res.status != Status::Success {
		let _ = send_message(pamh, &res.error, true);
		return PamResultCode::PAM_CRED_ERR;
	}

	PamResultCode::PAM_SUCCESS
}

fn send_message(pamh: &PamHandle, msg: &str, is_error: bool) -> Result<(), PamResultCode> {
	let conv = pamh.get_item::<Conv>()?.ok_or(PamResultCode::PAM_CONV_ERR)?;
	let style = if is_error { PAM_ERROR_MSG } else { PAM_TEXT_INFO };
	conv.send(style, msg).map(|_| ())
}

fn parse_args(args: Vec<&CStr>) -> BTreeMap<String, String> {
	args.into_iter()
		.filter_map(|arg| arg.to_str().ok())
		.map(|arg| {
			let mut parts = arg.splitn(2, '=');
			let key = parts.next().unwrap_or_default().to_owned();
			let value = parts.next().unwrap_or_default().to_owned();
			(key, value)
		})
		.collect()
}

fn render_user_template(template: &str, user: &users::User) -> String {
	let mut rendered = template.to_owned();
	let replacements = [
		("{{.Uid}}", user.uid().to_string()),
		("{{.Gid}}", user.primary_group_id().to_string()),
		("{{.Username}}", user.name().to_string_lossy().into_owned()),
		("{{.HomeDir}}", user.home_dir().to_string_lossy().into_owned()),
	];
	for (needle, replacement) in replacements {
		rendered = rendered.replace(needle, &replacement);
	}
	rendered
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::ffi::CString;

	#[test]
	fn parses_key_value_arguments() {
		let args = vec![
			CString::new("client=pam").unwrap(),
			CString::new("ifexist=/tmp/{{.Uid}}").unwrap(),
		];
		let refs = args.iter().map(|arg| arg.as_c_str()).collect::<Vec<_>>();
		let parsed = parse_args(refs);
		assert_eq!(parsed.get("client").map(String::as_str), Some("pam"));
		assert_eq!(parsed.get("ifexist").map(String::as_str), Some("/tmp/{{.Uid}}"));
	}
}
