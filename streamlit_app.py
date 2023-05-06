import streamlit as st
import subprocess
import os

st.title("Cohere-Parallel-Language-Sentence-Alignment")

# getting the API key
cohere_api_key = os.environ["COHERE_API_KEY"]

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
            "--cohere_api_key", cohere_api_key,
            "-m", "embed-multilingual-v2.0",
            "-s", "src.txt",
            "-t", "trg.txt",
            "-o", "cohere",
            "--retrieval", "nn",
            "--dot"
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        # Display the alignment output as a dataframe and a download link
        output_lines = result.stdout.strip().split("\n")
        output_data = [line.strip().split("\t") for line in output_lines]
        st.header("Alignment Output")
        st.dataframe(output_data)
        st.download_button(
            label="Download Output as TXT",
            data="\n".join(result.stdout),
            file_name="alignment_output.txt",
            mime="text/plain"
        )
