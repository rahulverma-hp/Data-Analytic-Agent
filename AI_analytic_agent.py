import tempfile
import csv
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools

load_dotenv()

def _first_env(*names: str) -> str:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return ""

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("Data Analytic Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    # Prefer OpenRouter key if present, else DeepSeek direct.
    env_api_key = _first_env("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    # If a key exists in .env, use it automatically but don't prefill the password input.
    if env_api_key and not st.session_state.get("api_key"):
        st.session_state.api_key = env_api_key

    api_key = st.text_input(
        "Enter your API key (OpenRouter or DeepSeek):",
        type="password",
        value=st.session_state.get("api_key", ""),
        placeholder=("Using key from .env" if env_api_key else "Paste your key here"),
    )
    if api_key:
        st.session_state.api_key = api_key
        st.success("API key loaded!")
    else:
        st.warning("Please enter your API key (or set OPENROUTER_API_KEY / DEEPSEEK_API_KEY in .env) to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and st.session_state.get("api_key"):
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Initialize DuckDbTools
        # Keep the tool surface minimal to improve compatibility with OpenRouter routing.
        duckdb_tools = DuckDbTools(
            include_tools=[
                "show_tables",
                "describe_table",
                "inspect_query",
                "run_query",
                "summarize_table",
                "export_table_to_path",
            ]
        )
        
        # Load the CSV file into DuckDB as a table
        duckdb_tools.load_local_csv_to_table(
            path=temp_path,
            table="uploaded_data",
        )
        
        # Initialize the Agent with DuckDB and Pandas tools
        # Provider selection:
        # - If OPENROUTER_API_KEY is set, use OpenRouter (and OPENROUTER_MODEL)
        # - Else use DeepSeek direct (DEEPSEEK_MODEL)
        using_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
        model_id = (
            _first_env("OPENROUTER_MODEL", "OPENAI_MODEL") if using_openrouter else _first_env("DEEPSEEK_MODEL", "OPENAI_MODEL")
        ) or ("deepseek/deepseek-chat" if using_openrouter else "deepseek-chat")
        base_url = (
            _first_env("OPENROUTER_BASE_URL", "OPENAI_BASE_URL") if using_openrouter else _first_env("DEEPSEEK_BASE_URL", "OPENAI_BASE_URL")
        ) or ("https://openrouter.ai/api/v1" if using_openrouter else "https://api.deepseek.com")
        max_tokens_env = (
            _first_env("OPENROUTER_MAX_TOKENS", "OPENAI_MAX_TOKENS")
            if using_openrouter
            else _first_env("DEEPSEEK_MAX_TOKENS", "OPENAI_MAX_TOKENS")
        )
        try:
            max_tokens = int(max_tokens_env) if max_tokens_env else None
        except ValueError:
            max_tokens = None
        openrouter_http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or None
        openrouter_app_title = os.getenv("OPENROUTER_APP_TITLE") or None
        default_headers = None
        if using_openrouter:
            default_headers = {
                k: v
                for k, v in {
                    "HTTP-Referer": openrouter_http_referer,
                    "X-Title": openrouter_app_title,
                }.items()
                if v
            } or None

        def build_agent(model: str) -> Agent:
            return Agent(
                model=OpenAIChat(
                    id=model,
                    api_key=st.session_state.api_key,
                    base_url=base_url,
                    default_headers=default_headers,
                    max_tokens=max_tokens,
                ),
                tools=[duckdb_tools],
                system_message="You are an expert data analyst. Use the 'uploaded_data' table to answer user queries. Generate SQL queries using DuckDB tools to solve the user's query. Provide clear and concise answers with the results.",
                markdown=True,
            )

        data_analyst_agent = build_agent(model_id)
        
        # Initialize code storage in session state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        response = data_analyst_agent.run(user_query)

                        # Extract the content from the response object
                        if hasattr(response, 'content'):
                            response_content = response.content
                        else:
                            response_content = str(response)

                    # Display the response in Streamlit
                    st.markdown(response_content)
                
                    
                except Exception as e:
                    st.error(f"Error generating response from the agent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")