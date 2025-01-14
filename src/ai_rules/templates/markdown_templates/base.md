# {assistant_name} 开发指南

{auto_generated_warning}

## 工具使用指南

### 工具调用格式
工具调用采用XML风格的标签格式，每个工具名称和参数都包含在标签内：

```xml
<tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
</tool_name>
```

### 可用工具
- execute_command：在系统上执行CLI命令
- read_file：读取指定文件内容
- write_to_file：创建或覆盖文件
- replace_in_file：对文件进行精确修改
- search_files：执行正则表达式搜索
- list_files：列出目录内容
- list_code_definition_names：列出代码定义
- use_mcp_tool：使用MCP服务器工具
- access_mcp_resource：访问MCP资源
- ask_followup_question：询问用户问题

## 环境配置
- 虚拟环境路径：`{venv_path}`
- 默认LLM提供商：`{default_provider}`
- 工作目录：`e:/chromeDownload/cline-main`
- 操作系统：Windows 10
- 默认Shell：cmd.exe

## 开发指南

### 代码风格
{code_style}

### 开发实践
{development}

### 项目管理
{project}

## LLM服务配置
可用的模型：
{llm_providers}

## MCP服务器

### weather服务器
工具：
- get-alerts：获取州的天气警报
- get-news：获取今日新闻
- get-forecast：获取天气预报

## 编辑规则
1. 使用replace_in_file进行小范围修改
2. 使用write_to_file创建新文件或完全重写
3. 确保提供完整行进行匹配
4. 按文件中的顺序列出多个修改
5. 等待用户确认每次修改

## 工作流程
1. 分析任务并设定目标
2. 逐步完成每个目标
3. 使用合适的工具
4. 等待用户确认
5. 完成后使用attempt_completion
