#[macro_use]
extern crate log;

use std::borrow::Cow;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai::{
    Client,
    types::{
        ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
};
use async_openai::config::OpenAIConfig;
use clap::Parser;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log::LevelFilter;
use rand::Rng;
use serde::Deserialize;
use strfmt::strfmt;
use warp::Filter;
use warp::http::StatusCode;
use warp::hyper::Body;
use warp::reply::Response;

#[derive(Parser)]
#[command(version)]
struct Args {
    config: PathBuf,
}

#[derive(Deserialize)]
struct Config {
    bind_addr: SocketAddr,
    system_template: String,
    system_with_style_template: String,
    api_keys: Vec<String>,
}

struct Context {
    openai_clients: Vec<Client<OpenAIConfig>>,
    system_template: String,
    system_with_style_template: String,
}

fn with_context(
    ctx: Arc<Context>
) -> impl Filter<Extract = (Arc<Context>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || ctx.clone())
}

#[derive(Deserialize)]
struct RichPromptReq {
    prompt: String,
    style: Option<String>
}

async fn rich_prompt(ctx: Arc<Context>, req: RichPromptReq) -> Response {
    let fut = async {
        let system = match req.style {
            None => Cow::Borrowed(ctx.system_template.as_str()),
            Some(style) => Cow::Owned(strfmt!(&ctx.system_with_style_template, style).map_err(|_| anyhow!("failed to format system_with_style_template"))?)
        };

        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(512u16)
            .model("gpt-3.5-turbo")
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system)
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(req.prompt)
                    .build()?
                    .into(),
            ])
            .top_p(0.0)
            .build()?;

        let clients = ctx.openai_clients.as_slice();
        let i = rand::thread_rng().gen_range(0..clients.len());
        let client = &clients[i];

        let mut response = client.chat().create(request).await?;
        let choice = response.choices.pop().ok_or_else(|| anyhow!("choices is empty"))?;
        let content = choice.message.content.ok_or_else(|| anyhow!("content is empty"))?;
        Result::<_, anyhow::Error>::Ok(content)
    };

    match fut.await {
        Ok(content) => {
            Response::new(Body::from(content))
        }
        Err(e) => {
            let mut resp = Response::new(Body::from(e.to_string()));
            *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            return resp;
        }
    }
}

fn logger_init() -> Result<()> {
    let pattern = if cfg!(debug_assertions) {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {f}:{L} - {m}{n}"
    } else {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {t} - {m}{n}"
    };

    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();

    let config = log4rs::Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(
            Root::builder()
                .appender("stdout")
                .build(LevelFilter::from_str(
                    &std::env::var("RICH_PROMPT_LOG").unwrap_or_else(|_| String::from("INFO")),
                )?),
        )?;

    log4rs::init_config(config)?;
    Ok(())
}

async fn serve(config: &Path) -> Result<()> {
    let c = tokio::fs::read_to_string(config).await?;
    let config: Config = toml::from_str(&c)?;
    let client = reqwest::Client::new();

    let mut clients = Vec::with_capacity(config.api_keys.len());

    for x in &config.api_keys {
        let open_ai_config = OpenAIConfig::new().with_api_key(x);
        let open_ai_client = Client::with_config(open_ai_config).with_http_client(client.clone());
        clients.push(open_ai_client);
    }

    let ctx = Context {
        openai_clients: clients,
        system_template: config.system_template,
        system_with_style_template: config.system_with_style_template
    };

    let ctx = Arc::new(ctx);

    let rich_prompt = warp::path!("richprompt")
        .and(warp::post())
        .and(with_context(ctx))
        .and(warp::body::json())
        .then(rich_prompt);

    let router = rich_prompt;
    let serve = warp::serve(router);
    info!("Listening on http://{}", config.bind_addr);
    serve.bind(config.bind_addr).await;
    Ok(())
}

fn main() -> ExitCode {
    let args: Args = Args::parse();
    let rt = tokio::runtime::Runtime::new().unwrap();
    logger_init().unwrap();

    match rt.block_on(serve(args.config.as_path())) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{:?}", e);
            ExitCode::FAILURE
        }
    }
}