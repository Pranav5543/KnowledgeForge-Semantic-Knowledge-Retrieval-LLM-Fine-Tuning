import gradio as gr
from ingestion import DocumentProcessor
from retrieval import PyramidRetriever

# Initialize global singletons to avoid re-loading weights
processor = DocumentProcessor()
retriever = PyramidRetriever()

def retrieve_answer(context, query):
    if not context.strip() or not query.strip():
        return "Please provide both context and a query."
        
    try:
        # 1. Process and Ingest the document dynamically
        pyramid = processor.ingest_document("user_doc", context)
        
        # 2. Perform Retrieval
        results = retriever.retrieve(query, pyramid, top_k=1)
        if not results:
            return "No relevant chunks found."
            
        best_match = results[0]
        
        output = f"### 🎯 Best Match Found!\n"
        output += f"**Similarity Score:** `{best_match['score']:.4f}`\n\n"
        output += f"**Detected Category:** `{best_match['category']}`\n\n"
        output += f"**Retrieved Summary:** {best_match['summary']}\n\n"
        output += f"**Raw Text Chunk (Snippet):**\n> {best_match['text'][:500]}..."
        
        return output
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"

# Default dummy data from the assignment script
dummy_text = (
    "The quick brown fox jumps over the lazy dog. This is a very standard test sentence. " * 50 +
    "Recently, the supreme court ruled that a contract must be signed by both parties to be legally binding. " * 30 +
    "Let's calculate the sum of the variables using a linear equation." * 30
)

# Build the Gradio App
with gr.Blocks(theme=gr.themes.Soft(), title="Vexoo Labs - Document Retrieval") as demo:
    gr.Markdown("# 🔍 Vexoo Labs AI Engineer Assignment - Retrieval Demo")
    gr.Markdown("This interface demonstrates the **4-Layer Knowledge Pyramid** Document Ingestion and **Semantic Retrieval** mechanisms.")
    
    with gr.Row():
        with gr.Column():
            context_input = gr.Textbox(
                label="Document Context (Paste your document here)", 
                lines=15, 
                value=dummy_text, 
                placeholder="Enter long text to ingest and structure into the knowledge pyramid..."
            )
            query_input = gr.Textbox(
                label="Semantic Search Query", 
                placeholder="e.g., What did the court say about contracts?"
            )
            submit_btn = gr.Button("Run Semantic Retrieval", variant="primary")
            
        with gr.Column():
            output_display = gr.Markdown("### Results will appear here...")
            
    submit_btn.click(
        fn=retrieve_answer, 
        inputs=[context_input, query_input], 
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch()
