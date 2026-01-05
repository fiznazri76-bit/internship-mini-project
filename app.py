import streamlit as st
import pandas as pd
from databricks import sql
import os
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Lex the Lead Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Database connection
@st.cache_resource
def get_databricks_connection():
    """Create Databricks SQL connection"""
    try:
        return sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
    except Exception as e:
        st.error(f"Failed to connect to Databricks: {str(e)}")
        return None

def query_lead_data(filters=None):
    """Query synthetic lead data with null exclusion"""
    conn = get_databricks_connection()
    if not conn:
        return None
    
    cursor = conn.cursor()
    
    # Query with NULL filters
    query = """
    SELECT 
        lead_id,
        Name,
        Email,
        Company,
        Title,
        Phone,
        Status,
        Rating,
        LeadSource,
        lead_created_date,
        LastActivityDate,
        IsConverted,
        lead_age_days,
        Funnel_Stage__c,
        MappedStatus__c,
        total_email_opens,
        email_opens_30d,
        email_opens_7d,
        total_email_clicks,
        email_clicks_30d,
        email_clicks_7d,
        total_forms_filled,
        forms_filled_30d,
        forms_filled_7d,
        unique_campaigns_engaged,
        last_marketo_activity_date,
        days_since_last_activity,
        engagement_temperature,
        engagement_score_30d,
        Owner_Role__c
    FROM user_development.sales_ai.synthetic_lead_data
    WHERE 1=1
        -- Exclude nulls in critical fields
        AND Name IS NOT NULL
        AND Email IS NOT NULL
        AND Company IS NOT NULL
        AND Title IS NOT NULL
        AND Status IS NOT NULL
        AND engagement_temperature IS NOT NULL
        AND engagement_score_30d IS NOT NULL
        
        -- Exclude converted leads
        AND (IsConverted IS NULL OR IsConverted = FALSE)
    """
    
    # Add user filters
    if filters:
        if filters.get('status'):
            status_list = ','.join([f"'{s}'" for s in filters['status']])
            query += f" AND Status IN ({status_list})"
        if filters.get('engagement_temp'):
            temp_list = ','.join([f"'{t}'" for t in filters['engagement_temp']])
            query += f" AND engagement_temperature IN ({temp_list})"
        if filters.get('min_score'):
            query += f" AND engagement_score_30d >= {filters['min_score']}"
    
    query += " ORDER BY engagement_score_30d DESC LIMIT 500"
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        cursor.close()
        return df
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        if cursor:
            cursor.close()
        return None

def call_openai_endpoint(prompt, system_message="You are Lex, an expert AI sales assistant for SDRs.", conversation_history=None):
    """Call OpenAI endpoint"""
    endpoint_name = os.getenv("OPENAI_ENDPOINT_NAME", "GTS-OpenAI")
    workspace_url = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    token = os.getenv("DATABRICKS_TOKEN")
    
    url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": system_message}]
    
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result:
            return result['choices'][0]['message']['content']
        elif 'predictions' in result:
            return result['predictions'][0]['content']
        else:
            return str(result)
            
    except Exception as e:
        return f"Error calling OpenAI endpoint: {str(e)}"

def generate_summary_stats(df):
    """Generate summary statistics"""
    stats = {
        "total_leads": len(df),
        "avg_engagement_score": df['engagement_score_30d'].mean() if 'engagement_score_30d' in df.columns else 0,
        "hot_leads": len(df[df['engagement_temperature'] == 'Hot']) if 'engagement_temperature' in df.columns else 0,
        "warm_leads": len(df[df['engagement_temperature'] == 'Warm']) if 'engagement_temperature' in df.columns else 0,
        "cold_leads": len(df[df['engagement_temperature'] == 'Cold']) if 'engagement_temperature' in df.columns else 0,
        "avg_email_opens_30d": df['email_opens_30d'].mean() if 'email_opens_30d' in df.columns else 0,
        "avg_email_clicks_30d": df['email_clicks_30d'].mean() if 'email_clicks_30d' in df.columns else 0,
        "avg_forms_filled_30d": df['forms_filled_30d'].mean() if 'forms_filled_30d' in df.columns else 0,
        "top_status": df['Status'].value_counts().head(3).to_dict() if 'Status' in df.columns else {},
        "avg_days_since_activity": df['days_since_last_activity'].mean() if 'days_since_last_activity' in df.columns else 0
    }
    return stats

def get_lead_context():
    """Generate context for chatbot"""
    if st.session_state.df is None or len(st.session_state.df) == 0:
        return "No lead data currently loaded."
    
    df = st.session_state.df
    stats = generate_summary_stats(df)
    
    context = f"""
Current lead data context:
- Total leads: {stats['total_leads']}
- Hot leads: {stats['hot_leads']}
- Warm leads: {stats['warm_leads']}
- Cold leads: {stats['cold_leads']}
- Average engagement score (30d): {stats['avg_engagement_score']:.1f}
- Average email opens (30d): {stats['avg_email_opens_30d']:.1f}
- Average email clicks (30d): {stats['avg_email_clicks_30d']:.1f}
- Average days since last activity: {stats['avg_days_since_activity']:.1f}
"""
    return context

# Sidebar - Lex Chatbot
with st.sidebar:
    st.markdown("## 🤖 Lex - Your AI Assistant")
    st.markdown("Ask me anything about your leads!")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask Lex about your leads...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        lead_context = get_lead_context()
        
        system_message = f"""You are Lex, an expert AI sales assistant for SDRs at Access Group. 
You help SDRs understand their leads, prioritize outreach, and provide actionable insights.

{lead_context}

Be conversational, helpful, and actionable. Keep responses concise (2-3 paragraphs max)."""
        
        with st.spinner("Lex is thinking..."):
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.chat_history[:-1]
            ]
            
            ai_response = call_openai_endpoint(
                user_input,
                system_message=system_message,
                conversation_history=conversation_history
            )
        
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Data controls
    st.markdown("### 📊 Data Controls")
    
    if st.button("🔄 Load Lead Data", type="primary", use_container_width=True):
        with st.spinner("Loading lead data..."):
            st.session_state.df = query_lead_data()
            st.session_state.data_loaded = True if st.session_state.df is not None else False
            if st.session_state.data_loaded:
                st.success(f"Loaded {len(st.session_state.df)} leads!")
    
    # Filters
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.markdown("### 🔍 Filters")
        df = st.session_state.df
        
        status_options = df['Status'].dropna().unique().tolist() if 'Status' in df.columns else []
        selected_status = st.multiselect("Lead Status", status_options, default=[])
        
        temp_options = ['Hot', 'Warm', 'Cool', 'Cold']
        selected_temp = st.multiselect("Engagement Temperature", temp_options, default=[])
        
        min_score = st.slider("Minimum Engagement Score (30d)", 0, 100, 0)
        
        if st.button("Apply Filters", use_container_width=True):
            filters = {
                'status': selected_status,
                'engagement_temp': selected_temp,
                'min_score': min_score
            }
            with st.spinner("Applying filters..."):
                st.session_state.df = query_lead_data(filters)
                st.success("Filters applied!")

# Main content
st.title("🤖 Lex the Lead Assistant")
st.markdown("AI-powered lead intelligence for SDRs (Demo with Synthetic Data)")

if not st.session_state.data_loaded:
    st.info("👈 Click 'Load Lead Data' in the sidebar to get started!")
    st.markdown("""
    ### Welcome to Lex! 🎯
    
    **What I can help you with:**
    - 🔥 Identify your hottest leads right now
    - 📊 Analyze engagement trends and patterns
    - 💡 Get personalized outreach recommendations
    - 🎯 Prioritize your call list for the day
    
    **Example questions:**
    - "Which leads should I call today?"
    - "Who are my top 5 hottest leads?"
    - "Show me leads from healthcare sector"
    
    Load your data and start chatting!
    """)
    st.stop()

if st.session_state.df is None or len(st.session_state.df) == 0:
    st.warning("No data available. Please check your filters or database connection.")
    st.stop()

df = st.session_state.df

# Key Metrics
st.subheader("📈 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Leads", f"{len(df):,}")

with col2:
    hot_leads = len(df[df['engagement_temperature'] == 'Hot']) if 'engagement_temperature' in df.columns else 0
    st.metric("🔥 Hot Leads", hot_leads)

with col3:
    avg_score = df['engagement_score_30d'].mean() if 'engagement_score_30d' in df.columns else 0
    st.metric("Avg Engagement Score", f"{avg_score:.1f}")

with col4:
    avg_opens = df['email_opens_30d'].mean() if 'email_opens_30d' in df.columns else 0
    st.metric("Avg Email Opens (30d)", f"{avg_opens:.1f}")

with col5:
    avg_clicks = df['email_clicks_30d'].mean() if 'email_clicks_30d' in df.columns else 0
    st.metric("Avg Email Clicks (30d)", f"{avg_clicks:.1f}")

# Quick AI Insights
st.subheader("🤖 Quick AI Insights")

col_ai1, col_ai2 = st.columns([2, 1])

with col_ai1:
    insight_type = st.selectbox(
        "Select Analysis Type",
        ["Overall Summary", "Hot Leads Analysis", "Engagement Trends", "Priority Recommendations"]
    )

with col_ai2:
    generate_button = st.button("Generate Insights", type="primary", use_container_width=True)

# Display insights outside of columns (full width)
if generate_button:
    with st.spinner("Lex is analyzing your data..."):
        stats = generate_summary_stats(df)
        
        prompts = {
            "Overall Summary": f"""Analyze this lead engagement data and provide a concise executive summary:
            
            Total Leads: {stats['total_leads']}
            Hot Leads: {stats['hot_leads']}
            Warm Leads: {stats['warm_leads']}
            Cold Leads: {stats['cold_leads']}
            Average Engagement Score: {stats['avg_engagement_score']:.1f}
            
            Provide 3-4 key insights and actionable recommendations.""",
            
            "Hot Leads Analysis": f"""Focus on the hot leads:
            
            Number of Hot Leads: {stats['hot_leads']}
            Average Email Opens (30d): {stats['avg_email_opens_30d']:.1f}
            Average Email Clicks (30d): {stats['avg_email_clicks_30d']:.1f}
            
            Provide specific strategies for SDRs to convert these hot leads.""",
            
            "Engagement Trends": f"""Analyze the engagement patterns:
            
            Hot: {stats['hot_leads']} leads
            Warm: {stats['warm_leads']} leads
            Cold: {stats['cold_leads']} leads
            
            What do these trends suggest about lead nurturing effectiveness?""",
            
            "Priority Recommendations": f"""Based on this data, provide prioritized action items:
            
            Total Leads: {stats['total_leads']}
            Hot Leads: {stats['hot_leads']}
            Average Engagement Score: {stats['avg_engagement_score']:.1f}
            
            List 5 specific, prioritized actions for the sales team."""
        }
        
        prompt = prompts.get(insight_type, prompts["Overall Summary"])
        insight = call_openai_endpoint(prompt)
        
        # Display insight in full width
        st.info(insight)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Lead Table", "📊 Engagement Analysis", "🎯 Top Performers", "📤 Export"])

with tab1:
    st.subheader("Lead Details")
    
    display_cols = [
        'Name', 'Email', 'Company', 'Title', 'Status', 'engagement_temperature',
        'engagement_score_30d', 'email_opens_30d', 'email_clicks_30d',
        'days_since_last_activity'
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    df_display = df[display_cols].copy()
    df_display = df_display.reset_index(drop=True)
    df_display.index = df_display.index + 1
    
    st.dataframe(
        df_display.style.background_gradient(
            subset=['engagement_score_30d'] if 'engagement_score_30d' in display_cols else [],
            cmap='RdYlGn'
        ),
        use_container_width=True,
        height=400
    )

with tab2:
    st.subheader("Engagement Distribution")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if 'engagement_temperature' in df.columns:
            temp_counts = df['engagement_temperature'].value_counts()
            st.bar_chart(temp_counts)
            st.caption("Leads by Engagement Temperature")
    
    with col_chart2:
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts().head(10)
            st.bar_chart(status_counts)
            st.caption("Top 10 Lead Statuses")
    
    if 'engagement_score_30d' in df.columns:
        st.subheader("Engagement Score Distribution")
        st.line_chart(df['engagement_score_30d'].value_counts().sort_index())

with tab3:
    st.subheader("Top Engaged Leads")
    
    if 'engagement_score_30d' in df.columns:
        top_leads = df.nlargest(20, 'engagement_score_30d').copy()
        
        display_cols_top = [
            'lead_id', 'Name', 'Email', 'Company', 'engagement_score_30d',
            'email_opens_30d', 'email_clicks_30d', 'forms_filled_30d'
        ]
        display_cols_top = [col for col in display_cols_top if col in top_leads.columns]
        
        top_leads_display = top_leads[display_cols_top].copy()
        top_leads_display = top_leads_display.reset_index(drop=True)
        top_leads_display.index = top_leads_display.index + 1
        
        st.dataframe(
            top_leads_display,
            use_container_width=True,
            height=500
        )

with tab4:
    st.subheader("Export Data")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"lex_leads_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        if 'engagement_score_30d' in df.columns:
            top_leads_export = df.nlargest(100, 'engagement_score_30d')
            csv_top = top_leads_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Top 100 Leads (CSV)",
                data=csv_top,
                file_name=f"lex_top_leads_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.info("💡 This is synthetic demo data - safe to export and share!")

# Footer
st.markdown("---")
st.markdown("**Lex the Lead Assistant** | Demo with Synthetic Data | Built on Databricks 🚀")
