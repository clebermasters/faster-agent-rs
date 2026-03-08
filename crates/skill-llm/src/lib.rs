use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use skill_tools::{ToolDefinition, ToolRegistry};
use std::collections::HashMap;
use std::pin::Pin;
use tracing::{debug, info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
}

pub trait LLMClient: Send + Sync {
    fn chat(&self, messages: Vec<Message>, tools: Option<Vec<ToolDefinition>>) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>>;
}

pub struct MiniMaxClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl MiniMaxClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
            model,
        }
    }
}

impl LLMClient for MiniMaxClient {
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        
        Box::pin(async move {
            // Convert system messages to user messages (MiniMax doesn't support system role)
            let messages: Vec<serde_json::Value> = messages
                .into_iter()
                .map(|m| {
                    if m.role == "system" {
                        // Convert system to user
                        json!({
                            "role": "user",
                            "content": m.content
                        })
                    } else if m.role == "tool" {
                        // MiniMax doesn't properly support tool_call_id - send as regular message
                        json!({
                            "role": "user",
                            "content": format!("[Tool Result for {}]: {}", m.tool_call_id.unwrap_or_default(), m.content)
                        })
                    } else {
                        json!({
                            "role": m.role,
                            "content": m.content
                        })
                    }
                })
                .collect();

            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": false,
            });

        if let Some(tools) = tools {
            let openai_tools: Vec<serde_json::Value> = tools
                .into_iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(openai_tools);
        }

        debug!("MiniMax request body: {}", body);

        debug!("Sending chat request to MiniMax: {}", base_url);

        let response = client
            .post(format!("{}/v1/chat/completions", base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("MiniMax API error: {} - {}", status, error_text);
        }

        let chat_resp: serde_json::Value = response.json().await?;

        let message = chat_resp["choices"][0]["message"].clone();
        let role = message["role"]
            .as_str()
            .unwrap_or("assistant")
            .to_string();
        let content = message["content"].as_str().unwrap_or("").to_string();

        let tool_calls = if message.get("tool_calls").is_some() {
            let calls: Vec<ToolCall> = message["tool_calls"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|c| {
                    let func = c.get("function")?;
                    Some(ToolCall {
                        id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        name: func["name"].as_str()?.to_string(),
                        arguments: func["arguments"].clone(),
                    })
                })
                .collect();
            if calls.is_empty() {
                None
            } else {
                Some(calls)
            }
        } else {
            None
        };

        Ok(ChatResponse {
            message: Message { role, content, tool_call_id: None },
            tool_calls,
            done: true,
        })
        })
    }
}

pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<ChatResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        if let Some(tools) = tools {
            let ollama_tools: Vec<serde_json::Value> = tools
                .into_iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(ollama_tools);
        }

        debug!("Sending chat request to Ollama");

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error: {} - {}", status, error_text);
        }

        let chat_resp: serde_json::Value = response.json().await?;

        let message = chat_resp["message"].clone();
        let role = message["role"]
            .as_str()
            .unwrap_or("assistant")
            .to_string();
        let content = message["content"].as_str().unwrap_or("").to_string();

        let tool_calls = if message.get("tool_calls").is_some() {
            let calls: Vec<ToolCall> = message["tool_calls"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|c| {
                    let func = c.get("function")?;
                    Some(ToolCall {
                        id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        name: func["name"].as_str()?.to_string(),
                        arguments: func["arguments"].clone(),
                    })
                })
                .collect();
            if calls.is_empty() {
                None
            } else {
                Some(calls)
            }
        } else {
            None
        };

        Ok(ChatResponse {
            message: Message { role, content, tool_call_id: None },
            tool_calls,
            done: chat_resp["done"].as_bool().unwrap_or(true),
        })
    }

    pub async fn generate(&self, prompt: String) -> Result<String> {
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        });

        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error: {} - {}", status, error_text);
        }

        let resp: serde_json::Value = response.json().await?;
        Ok(resp["response"].as_str().unwrap_or("").to_string())
    }
}

impl LLMClient for OllamaClient {
    fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse>> + Send + '_>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let model = self.model.clone();
        
        Box::pin(async move {
            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": false,
            });

            if let Some(tools) = tools {
                let ollama_tools: Vec<serde_json::Value> = tools
                    .into_iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::Value::Array(ollama_tools);
            }

            debug!("Sending chat request to Ollama");

            let response = client
                .post(format!("{}/api/chat", base_url))
                .json(&body)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                anyhow::bail!("Ollama API error: {} - {}", status, error_text);
            }

            let chat_resp: serde_json::Value = response.json().await?;

            let message = chat_resp["message"].clone();
            let role = message["role"]
                .as_str()
                .unwrap_or("assistant")
                .to_string();
            let content = message["content"].as_str().unwrap_or("").to_string();

        let tool_calls = if message.get("tool_calls").is_some() {
            let calls: Vec<ToolCall> = message["tool_calls"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|c| {
                    let func = c.get("function")?;
                    Some(ToolCall {
                        id: c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        name: func["name"].as_str()?.to_string(),
                        arguments: func["arguments"].clone(),
                    })
                })
                .collect();
            if calls.is_empty() {
                None
            } else {
                Some(calls)
            }
        } else {
            None
        };

            Ok(ChatResponse {
                message: Message { role, content, tool_call_id: None },
                tool_calls,
                done: chat_resp["done"].as_bool().unwrap_or(true),
            })
        })
    }
}

pub struct Agent {
    llm: Box<dyn LLMClient>,
    tool_registry: ToolRegistry,
    max_iterations: usize,
    tool_call_history: HashMap<String, usize>,
}

impl Agent {
    pub fn new(llm: Box<dyn LLMClient>) -> Self {
        Self {
            llm,
            tool_registry: ToolRegistry::new(),
            max_iterations: 10,
            tool_call_history: HashMap::new(),
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tool_registry = tools;
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub async fn run(&self, task: &str) -> Result<String> {
        let system_prompt = r#"You are an autonomous agent that MUST use tools to complete tasks.

AVAILABLE TOOLS:
- bash: Execute shell commands
- read: Read files  
- write: Write/save content to files (THIS IS HOW YOU SAVE SCRAPED CONTENT)
- skill_web-scraper: Returns scraped HTML content
- skill_rss-fetcher: Returns RSS feed content  
- skill_code-review: Reviews code

STRICT RULES:
1. If you get content from skill_web-scraper, you MUST call 'write' to save it
2. If you get content from skill_rss-fetcher, you MUST call 'write' to save it
3. NEVER just output content - you MUST save it to a file with 'write'
4. The task is NOT done until content is saved to a file

Example - "scrape example.com save to file.html":
- Step 1: Call skill_web-scraper with URL
- Step 2: Copy the scraped content from result
- Step 3: Call write with path="file.html" and content="<paste scraped content>"
- Step 4: Task complete

The 'write' tool is REQUIRED to save any content. It is your job to save things!"#;

        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: task.to_string(),
                tool_call_id: None,
            },
        ];

        let tool_defs = self.tool_registry.list();
        let mut tool_history: HashMap<String, usize> = HashMap::new();

        info!("Starting agent loop for task: {}", task);
        info!("Available tools: {:?}", self.tool_registry.names());
        debug!("Total messages at start: {}", messages.len());

        for iteration in 0..self.max_iterations {
            info!("=== Iteration {}/{} ===", iteration + 1, self.max_iterations);
            debug!("Messages before LLM call: {}", messages.len());
            
            // Log last few messages for debugging
            if iteration > 0 {
                debug!("Last 3 messages roles: {:?}", messages.iter().rev().take(3).map(|m| &m.role).collect::<Vec<_>>());
            }

            let response = self.llm.chat(messages.clone(), Some(tool_defs.clone())).await?;
            
        debug!("LLM response - has_tool_calls: {}, content_len: {}", 
            response.tool_calls.is_some(), 
            response.message.content.len());
        
        // Debug: log raw response if empty
        if response.message.content.is_empty() && response.tool_calls.is_none() {
            warn!("LLM returned empty response (no content, no tool_calls)! Message: {:?}", response.message);
        }

            if let Some(tool_calls) = response.tool_calls {
                info!("LLM returned {} tool call(s)", tool_calls.len());
                
                // Execute tool calls ONE AT A TIME and ask LLM for next step after each
                // This enables chaining - LLM sees result before deciding next action
                for (i, call) in tool_calls.iter().enumerate() {
                    info!("  -> Tool {}: {} args: {:?}", i+1, call.name, call.arguments);
                    
                    let call_key = format!("{}:{}", call.name, call.arguments);
                    let count = tool_history.entry(call_key.clone()).or_insert(0);
                    *count += 1;
                    debug!("Tool '{}' call count: {}", call.name, count);

                    if *count > 2 {
                        warn!("Doom loop detected: {} called {} times", call.name, count);
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: format!(
                                "ERROR: Tool '{}' has failed multiple times ({} attempts). \
                                Try a DIFFERENT approach.",
                                call.name, count
                            ),
                            tool_call_id: None,
                        });
                        continue;
                    }

                    if let Some(tool) = self.tool_registry.get(&call.name) {
                        info!("Executing tool: {}", call.name);
                        
                        // Handle arguments - MiniMax sends them as a string, need to parse
                        let args = if let Some(s) = call.arguments.as_str() {
                            serde_json::from_str(s).unwrap_or(call.arguments.clone())
                        } else {
                            call.arguments.clone()
                        };
                        
                        let result = tool.execute(args).await?;
                        
                        info!("Tool {} result: success={}, output_len={}", 
                            call.name, result.success, result.output.len());

                        // If write tool succeeded, task is complete
                        if call.name == "write" && result.success {
                            info!("Write tool succeeded - task complete!");
                            return Ok(format!(
                                "Task completed successfully! Saved content to: {}",
                                call.arguments.get("path")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("file")
                            ));
                        }

                        let result_message = Self::format_tool_result(&call.name, &result);
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: result_message,
                            tool_call_id: call.id.clone(),
                        });
                        
                        // CRITICAL: Explicitly ask LLM what to do next
                        messages.push(Message {
                            role: "system".to_string(),
                            content: format!(
                                "You just called '{}' and got a result. \n\
                                Your task is: {}\n\
                                What is the NEXT step? Do you need to call another tool? \
                                If you got content from a skill, you MUST save it with 'write'.",
                                call.name, task
                            ),
                            tool_call_id: None,
                        });
                        
                        debug!("Asking LLM what to do next after tool: {}", call.name);
                        break; // Exit the for loop to ask LLM for next step
                    } else {
                        error!("Tool not found: {}", call.name);
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: format!("ERROR: Tool '{}' not found.", call.name),
                            tool_call_id: None,
                        });
                    }
                }

                continue;
            } else {
                let final_response = response.message.content.clone();
                debug!("No tool calls, LLM responded directly. Response: {:?}", &final_response[..final_response.len().min(200)]);
                
                if final_response.trim().is_empty() {
                    // Empty response - prompt the LLM to try again
                    info!("LLM returned empty response, prompting to continue...");
                    messages.push(Message {
                        role: "system".to_string(),
                        content: "You MUST use a tool to complete the task. If you got content from a skill, use the 'write' tool to save it. What tool will you call next?".to_string(),
                        tool_call_id: None,
                    });
                    continue;
                }
                
                // Check if this looks like a final answer (not asking for more tools)
                let lower_response = final_response.to_lowercase();
                let is_final = !lower_response.contains("tool") && 
                               !lower_response.contains("call") && 
                               !lower_response.contains("need to") &&
                               !lower_response.contains("should i") &&
                               !lower_response.contains("would you");
                
                if is_final || lower_response.len() > 50 {
                    // This looks like a final answer - return it
                    info!("Detected final answer from LLM");
                    return Ok(final_response);
                }
                
                // LLM is still trying to use tools or asking questions
                // Try one more time with a stronger prompt
                info!("LLM returned text but no tool calls. Prompting to use tools...");
                messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Your previous response '{}' didn't use any tools. The task '{}' requires using tools. What is the next tool you need to call?",
                        final_response.chars().take(100).collect::<String>(), task
                    ),
                    tool_call_id: None,
                });
            }
        }

        warn!("Max iterations reached without completing task");
        Ok("Max iterations reached. Task may not be complete.".to_string())
    }

    fn format_tool_result(tool_name: &str, result: &skill_tools::ToolResult) -> String {
        if result.success {
            if result.output.is_empty() {
                format!("Tool '{}' completed successfully but produced no output.", tool_name)
            } else {
                format!("Tool '{}' succeeded:\n{}", tool_name, result.output)
            }
        } else {
            let error_msg = result.error.as_deref().unwrap_or("Unknown error");
            format!(
                "Tool '{}' FAILED with error: {}\n\nPrevious output: {}",
                tool_name, error_msg, result.output
            )
        }
    }
}
