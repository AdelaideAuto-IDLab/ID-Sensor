//! A stripped down version of the `training_manager` tool for the ID-Sensor project

#![feature(proc_macro_hygiene, decl_macro)]

mod batch_gen;
mod config;
mod data;
mod task;
mod types;
mod utils;

use std::io::BufRead;
use std::{env, error::Error, fs, io};

use rocket::{catch, catchers, delete, get, post, routes};
use rocket_contrib::json;
use rocket_contrib::json::{Json, JsonValue};

use crate::config::GenSettings;

fn main() {
    env::set_var("ROCKET_CLI_COLORS", "0");
    rocket().launch();
}

fn rocket() -> rocket::Rocket {
    let routes = routes![
        get_results,
        list_tasks,
        delete_all,
        create_task,
        delete_task,
        task_info,
        get_batch
    ];

    rocket::ignite()
        .mount("/", routes)
        .register(catchers![not_found])
}

type JsonResult<T> = Result<Json<T>, Box<dyn Error>>;

#[catch(404)]
fn not_found() -> JsonValue {
    json!({
        "status": "error",
        "reason": "Resource was not found."
    })
}

#[get("/results")]
fn get_results() -> JsonResult<Vec<serde_json::Value>> {
    match env::var("RESULT_PATH") {
        Ok(path) => {
            let mut output = vec![];

            let mut reader = io::BufReader::new(fs::File::open(path)?);
            let mut buffer = String::new();
            while reader.read_line(&mut buffer)? != 0 {
                output.push(serde_json::from_str(&buffer)?);
                buffer.clear();
            }

            Ok(Json(output))
        }
        Err(_) => Err("`RESULT_PATH` is not set".into()),
    }
}

#[get("/task")]
fn list_tasks() -> Json<Vec<task::TaskInfo>> {
    Json(task::get_manager().read().list())
}

#[delete("/task")]
fn delete_all() -> Json<usize> {
    Json(task::get_manager().write().delete_all())
}

#[post("/task/<id>", data = "<settings>")]
fn create_task(id: String, settings: Json<GenSettings>) -> JsonResult<task::TaskInfo> {
    let info = task::get_manager().write().new_task(id, settings.0)?;
    Ok(Json(info))
}

#[delete("/task/<id>")]
fn delete_task(id: String) -> Json<bool> {
    Json(task::get_manager().write().delete_task(&id))
}

#[get("/task/<id>")]
fn task_info(id: String) -> Json<Option<task::TaskInfo>> {
    Json(task::get_manager().read().get_task(&id))
}

#[get("/task/<id>/batch/<step>")]
fn get_batch(id: String, step: usize) -> JsonResult<types::Batch> {
    task::get_manager().write().update_task(&id, step);
    let batch = task::get_manager().read().get_batch(&id, step)?;
    Ok(Json(batch))
}
