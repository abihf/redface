use std::io::Write;

use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
	#[serde(rename = "AUTH")]
	Authenticate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Req {
	pub action: Action,
	pub params: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct AuthReq {
	pub client: String,
	pub user: String,
	pub timeout: Option<i32>,
	pub show_osd: bool,
}

impl AuthReq {
	pub fn write_to<W: Write>(&self, mut writer: W) -> serde_json::Result<()> {
		let mut req = Req {
			action: Action::Authenticate,
			params: std::collections::BTreeMap::from([
				("client".to_owned(), self.client.to_owned()),
				("user".to_owned(), self.user.to_owned()),
			]),
		};
		if let Some(timeout) = self.timeout {
			req.params.insert("timeout".to_owned(), timeout.to_string());
		}
		serde_json::to_writer(&mut writer, &req)?;
		writer.write_all(b"\n").map_err(serde_json::Error::io)
	}
}

impl From<Req> for AuthReq {
	fn from(req: Req) -> Self {
		let timeout = req.params.get("timeout").and_then(|value| value.parse::<i32>().ok());
		let show_osd = req.params.get("show_osd").map(|value| value == "true").unwrap_or(false);
		Self {
			client: req.params.get("client").cloned().unwrap_or_default(),
			user: req.params.get("user").cloned().unwrap_or_default(),
			timeout,
			show_osd,
		}
	}
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
	#[serde(rename = "SUCCESS")]
	Success,
	#[serde(rename = "ERROR")]
	Error,
}

impl Status {
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::Success => "SUCCESS",
			Self::Error => "ERROR",
		}
	}
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Res {
	pub status: Status,
	#[serde(default)]
	pub error: String,
	#[serde(default)]
	pub extras: std::collections::BTreeMap<String, String>,
}

pub fn write_success_res(
	mut writer: impl Write,
	extras: std::collections::BTreeMap<String, String>,
) -> serde_json::Result<()> {
	let res = Res {
		status: Status::Success,
		error: String::new(),
		extras,
	};
	serde_json::to_writer(&mut writer, &res)?;
	writer.write_all(b"\n").map_err(serde_json::Error::io)
}

pub fn write_error_res(mut writer: impl Write, err: &dyn std::error::Error) -> serde_json::Result<()> {
	let res = Res {
		status: Status::Error,
		error: err.to_string(),
		extras: std::collections::BTreeMap::new(),
	};
	serde_json::to_writer(&mut writer, &res)?;
	writer.write_all(b"\n").map_err(serde_json::Error::io)
}
