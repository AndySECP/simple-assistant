import streamlit as st
import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

class ThoughtfulAISupport:
    def __init__(self):
        # load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)   
        # load JSON prededined data
        self.load_predefined_data()
        self.similarity_threshold = 0.7
        self.question_embeddings = None
        
        # initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def load_predefined_data(self):
        """Load the predefined question-answer data from JSON file"""
        with open("predefined_data.json", 'r') as f:
            self.predefined_data = json.load(f)

    @st.cache_data
    def get_question_embeddings(_self):
        """Generate and cache embeddings for all predefined questions"""
        questions = [q["question"] for q in _self.predefined_data["questions"]]
        embeddings = []
        
        for question in questions:
            response = _self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=question
            )
            embeddings.append(response.data[0].embedding)
        
        return embeddings

    def find_best_match(self, query, threshold=None):
        """Find the best matching predefined question using embeddings"""
        if threshold is None:
            threshold = self.similarity_threshold
            
        if self.question_embeddings is None:
            # Fall back to keyword matching if embeddings aren't available
            return self.find_best_match_keywords(query)
        
        try:
            # Get embedding for the query
            query_response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = query_response.data[0].embedding
            
            # Calculate cosine similarity with all predefined questions
            similarities = []
            for emb in self.question_embeddings:
                similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                similarities.append(similarity)
            
            # Find the most similar question
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity >= threshold:
                return {
                    "found": True, 
                    "qa": self.predefined_data["questions"][max_similarity_idx], 
                    "score": max_similarity
                }
            else:
                return {"found": False, "score": max_similarity}
        except Exception as e:
            st.sidebar.warning(f"Error in embedding matching: {str(e)}")
            # Fall back to keyword matching if embedding fails
            return self.find_best_match_keywords(query)

    def find_best_match_keywords(self, query):
        """Simple keyword matching as fallback method"""
        query = query.lower()
        best_score = 0
        best_qa = None
        
        for qa in self.predefined_data["questions"]:
            question = qa["question"].lower()
            # Count how many words from the query appear in the question
            words = set(query.split())
            matches = sum(1 for word in words if word in question)
            score = matches / len(words) if words else 0
            
            if score > best_score:
                best_score = score
                best_qa = qa
        
        if best_score > 0.3:  # Simple threshold
            return {"found": True, "qa": best_qa, "score": best_score}
        return {"found": False, "score": best_score}

    def generate_streaming_response(self, query, context=None):
        """Generate a streaming response using OpenAI"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful customer support agent for Thoughtful AI. You should use the provided information to answer questions about Thoughtful AI and respond naturally to other queries."}
            ]
            
            # Add context if available
            if context and context["found"]:
                messages.append({
                    "role": "system", 
                    "content": f"Here is relevant information to help answer the question: {context['qa']['question']} - {context['qa']['answer']}"
                })
            
            messages.append({"role": "user", "content": query})
            
            stream = self.client.chat.completions.create(
                model="gpt-4-turbo-preview", 
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                stream=True
            )
            
            return stream
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def setup_ui(self):
        """Set up the Streamlit UI components"""
        st.title("Thoughtful AI Customer Support")
        st.markdown("*Ask me anything about Thoughtful AI's agents or general questions.*")
        
        # Sidebar setup
        with st.sidebar:
            st.header("Our AI Agents")
            st.markdown("- **EVA**: Eligibility Verification Agent")
            st.markdown("- **CAM**: Claims Processing Agent")
            st.markdown("- **PHIL**: Payment Posting Agent")
            
            self.similarity_threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.5, 
                max_value=0.9, 
                value=0.7, 
                step=0.05,
                help="Minimum similarity score required to use a predefined answer"
            )
            
            st.divider()
            st.markdown("*This is a demo customer support agent for Thoughtful AI.*")
        
        # Load embeddings for predefined questions
        try:
            self.question_embeddings = self.get_question_embeddings()
            st.sidebar.success("Embeddings loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading embeddings: {str(e)}")
            # Embeddings will remain None, and we'll use keyword fallback

    def display_chat_history(self):
        """Display the chat history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        """Process user input and generate response"""
        user_query = st.chat_input("Ask me about Thoughtful AI")
        
        if user_query:
            # display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # find the best match
            match_result = self.find_best_match(user_query)
            
            if match_result["found"]:
                st.sidebar.info(f"Found matching question with score: {match_result['score']:.2f}")
                st.sidebar.info(f"Matched: '{match_result['qa']['question']}'")
            else:
                st.sidebar.info(f"No good match found. Highest similarity: {match_result['score']:.2f}")
            
            # display the assistant response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                stream = self.generate_streaming_response(user_query, context=match_result)
                
                # check if we got an error string instead of a stream
                if isinstance(stream, str):
                    message_placeholder.markdown(stream)
                    st.session_state.messages.append({"role": "assistant", "content": stream})
                else:
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # add the response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

    def run(self):
        """Main method to run the Streamlit app"""
        self.setup_ui()
        self.display_chat_history()
        self.handle_user_input()
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: gray;'>Powered by Thoughtful AI</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    app = ThoughtfulAISupport()
    app.run()
