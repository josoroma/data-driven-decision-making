setup:
	poetry install
	poetry env use `which python3.11`
	poetry shell

add:
	poetry add streamlit streamlit_chat streamlit_extras langchain openai tiktoken pypdf pinecone-client

run:
	poetry run streamlit run app.py

clean:
	rm -rf `poetry env info -p`
	rm -rf poetry.lock
