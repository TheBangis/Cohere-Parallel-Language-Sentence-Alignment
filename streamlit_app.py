import streamlit as st
import subprocess

st.title("Cohere-Parallel-Language-Sentence-Alignment Demo")

# Upload source and target files
src_file = st.file_uploader("Upload source file", type=["txt"])
trg_file = st.file_uploader("Upload target file", type=["txt"])

# Run the aligner and display the output
if st.button("Align"):
    if src_file is None or trg_file is None:
        st.warning("Please upload both source and target files.")
    else:
        # Save the uploaded files
        with open("src.txt", "wb") as f:
            f.write(src_file.read())
        with open("trg.txt", "wb") as f:
            f.write(trg_file.read())
        
        # Run the aligner command
        command = [
            "python3",
            "scripts/cohere_align.py",
            "--cohere_api_key", "<api_key>",
            "-m", "embed-multilingual-v2.0",
            "-s", "src.txt",
            "-t", "trg.txt",
            "-o", "cohere",
            "--retrieval", "nn",
            "--dot",
            "--cuda"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Display the alignment output
        st.header("Alignment Output")
        st.text(result.stdout)
