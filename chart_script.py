import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Data from the provided JSON
components = [
    {
        "name": "User Interface",
        "type": "interface",
        "subtypes": ["Terminal Interface", "Web Interface (Flask)"],
        "inputs": ["User Messages"],
        "outputs": ["Bot Responses", "Analytics Display"]
    },
    {
        "name": "ChatBot Core",
        "type": "core",
        "responsibilities": ["Message Processing", "Session Management", "Response Generation"],
        "inputs": ["Raw User Input"],
        "outputs": ["Formatted Responses"]
    },
    {
        "name": "Text Processor",
        "type": "nlp",
        "features": ["Tokenization", "Stopword Removal", "Lemmatization", "Text Cleaning"],
        "inputs": ["Raw Text"],
        "outputs": ["Processed Tokens"]
    },
    {
        "name": "Intent Classifier",
        "type": "ai",
        "methods": ["Keyword Matching", "Similarity Scoring", "Pattern Recognition"],
        "inputs": ["Processed Text"],
        "outputs": ["Intent + Confidence Score"]
    },
    {
        "name": "Conversation Logger",
        "type": "storage",
        "features": ["JSON Logging", "Analytics Generation", "Session Tracking"],
        "inputs": ["User Input + Bot Response + Metadata"],
        "outputs": ["Conversation Analytics"]
    },
    {
        "name": "Extension Framework",
        "type": "extensible",
        "extensions": ["Sentiment Analysis", "ML Classification", "Context Management", "API Integration"],
        "status": "Ready for AI/ML Integration"
    }
]

data_flow = [
    {"from": "User Interface", "to": "ChatBot Core", "data": "User Message"},
    {"from": "ChatBot Core", "to": "Text Processor", "data": "Raw Text"},
    {"from": "Text Processor", "to": "Intent Classifier", "data": "Processed Tokens"},
    {"from": "Intent Classifier", "to": "ChatBot Core", "data": "Intent + Response"},
    {"from": "ChatBot Core", "to": "Conversation Logger", "data": "Conversation Data"},
    {"from": "ChatBot Core", "to": "User Interface", "data": "Bot Response"},
    {"from": "Extension Framework", "to": "ChatBot Core", "data": "Enhanced Processing"}
]

# Create component positions in a logical layout
positions = {
    "User Interface": (0, 2),
    "ChatBot Core": (2, 2),
    "Text Processor": (4, 3),
    "Intent Classifier": (4, 1),
    "Conversation Logger": (2, 0),
    "Extension Framework": (0, 0)
}

# Define colors for different component types
type_colors = {
    "interface": "#1FB8CD",
    "core": "#DB4545", 
    "nlp": "#2E8B57",
    "ai": "#5D878F",
    "storage": "#D2BA4C",
    "extensible": "#B4413C"
}

# Create the figure
fig = go.Figure()

# Add rectangular shapes for components
for comp in components:
    pos = positions[comp["name"]]
    color = type_colors[comp["type"]]
    
    # Add rectangle shape
    fig.add_shape(
        type="rect",
        x0=pos[0] - 0.7,
        y0=pos[1] - 0.4,
        x1=pos[0] + 0.7,
        y1=pos[1] + 0.4,
        fillcolor=color,
        line=dict(color="white", width=3),
        layer="below"
    )
    
    # Create hover text with more details
    hover_text = f"<b>{comp['name']}</b><br>"
    hover_text += f"Type: {comp['type']}<br><br>"
    
    if "subtypes" in comp:
        hover_text += f"Subtypes:<br>" + "<br>".join([f"• {st}" for st in comp["subtypes"]]) + "<br><br>"
    if "responsibilities" in comp:
        hover_text += f"Functions:<br>" + "<br>".join([f"• {resp}" for resp in comp["responsibilities"]]) + "<br><br>"
    if "features" in comp:
        hover_text += f"Features:<br>" + "<br>".join([f"• {feat}" for feat in comp["features"]]) + "<br><br>"
    if "methods" in comp:
        hover_text += f"Methods:<br>" + "<br>".join([f"• {method}" for method in comp["methods"]]) + "<br><br>"
    if "extensions" in comp:
        hover_text += f"Extensions:<br>" + "<br>".join([f"• {ext}" for ext in comp["extensions"]]) + "<br><br>"
    
    if "inputs" in comp:
        hover_text += f"Inputs: {', '.join(comp['inputs'])}<br>"
    if "outputs" in comp:
        hover_text += f"Outputs: {', '.join(comp['outputs'])}"
    
    # Add invisible scatter point for hover and text
    fig.add_trace(go.Scatter(
        x=[pos[0]],
        y=[pos[1]],
        mode='text',
        text=comp["name"],
        textposition="middle center",
        textfont=dict(size=12, color='white', family="Arial Black"),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

# Add arrows with labels
for flow in data_flow:
    from_pos = positions[flow["from"]]
    to_pos = positions[flow["to"]]
    
    # Calculate arrow positions (from edge of box to edge of box)
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Start from edge of source box
        start_x = from_pos[0] + dx_norm * 0.7
        start_y = from_pos[1] + dy_norm * 0.4
        
        # End at edge of target box
        end_x = to_pos[0] - dx_norm * 0.7
        end_y = to_pos[1] - dy_norm * 0.4
        
        # Add arrow line
        fig.add_trace(go.Scatter(
            x=[start_x, end_x],
            y=[start_y, end_y],
            mode='lines',
            line=dict(color='#333333', width=3),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add arrowhead
        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=start_x,
            ay=start_y,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor='#333333',
            arrowwidth=3,
            showarrow=True
        )
        
        # Add data flow label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Offset label to avoid overlapping with arrow
        offset_x = -dy_norm * 0.3 if abs(dy_norm) > 0.1 else 0.3
        offset_y = dx_norm * 0.3 if abs(dx_norm) > 0.1 else 0.3
        
        fig.add_trace(go.Scatter(
            x=[mid_x + offset_x],
            y=[mid_y + offset_y],
            mode='text',
            text=flow["data"][:15],  # Limit to 15 chars as per instructions
            textfont=dict(size=10, color='#333333', family="Arial"),
            textposition="middle center",
            hoverinfo='none',
            showlegend=False
        ))

# Update layout
fig.update_layout(
    title=dict(
        text="Chatbot System Architecture",
        font=dict(size=20),
        x=0.5,
        xanchor='center'
    ),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-1, 5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.8, 3.8]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("chatbot_architecture.png", width=1200, height=800)