import requests
import json
import re
import sys
class LLMOutputCleaner:
    """
    A class to extract and clean text from LLM outputs wrapped in markdown code blocks.
    
    Handles:
    - Extraction of content between ``` markers.
    - Removal of language identifier after opening ``` (e.g., 'python', 'json').
    - Replacement of literal '\\n' sequences with actual newlines.
    """

    def __init__(self):
        # Regex to match the content inside triple backticks.
        # It captures everything between the first ``` and the next ```.
        # It allows for optional whitespace and language tags after the opening ```.
        self.code_block_pattern = re.compile(
            r'```(?:\w+)?\s*\n(.*?)\n?```',
            re.DOTALL
        )

    def extract_and_clean(self, raw_output: str) -> str:
        """
        Extracts the code block content, removes the language tag,
        and replaces literal '\\n' with actual newlines.
        
        Args:
            raw_output (str): The raw string returned by the LLM.
            
        Returns:
            str: The cleaned content with actual newlines.
        """
        if not raw_output:
            return ""

        # Step 1: Try to find the code block
        match = self.code_block_pattern.search(raw_output)
        
        if match:
            # Extract the content group (group 1)
            content = match.group(1)
        else:
            # Fallback: If no code block found, assume the whole string is the content
            # This is a safety net, but you might want to raise an error instead.
            content = raw_output.strip()

        # Step 2: Clean up the content
        # Replace literal backslash-n (\n) with actual newline characters
        # Note: In Python strings, '\\n' represents the two characters '\' and 'n'
        cleaned_content = content.replace('\\n', '\n')

        # Optional: Strip leading/trailing whitespace if desired
        # cleaned_content = cleaned_content.strip()

        return cleaned_content

    def extract_only(self, raw_output: str) -> str:
        """
        Extracts the raw content without replacing literal \n.
        Useful if you want to inspect the raw extracted text.
        """
        if not raw_output:
            return ""
        
        match = self.code_block_pattern.search(raw_output)
        if match:
            return match.group(1)
        else:
            return raw_output.strip()
cleaner = LLMOutputCleaner()


def main():
    # Get user input
    session_open = input("Open Session: ")
    context = ""
    system_prompt = ""
    if session_open != "":
        with open(f"{session_open}.system", "r") as f:
            system_prompt = f.read()
    if system_prompt == "":
        system_prompt = input("Enter system prompt: ")
    if session_open != "":
        with open(f"{session_open}.memory", "r") as f:
            context = f.read()
    while(1):
        user_message = input("Enter your message: ")
        if user_message == ':q':
            sys.exit(0)
        if user_message == ':w':
            session_name = input("Enter Session Name (File): ")
            with open(f"{session_name}.system", "w", encoding="utf-8") as f:
                f.write("This is the first line.\n")
            with open(f"{session_name}.memory", "w", encoding="utf-8") as f:
                f.write(context)
        #print(f"{context}")
        # Prepare the payload
        payload = {
            "prompt": user_message,
            "system": system_prompt,
            "context": context,
            "stream": True
        }

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*"
        }

        # Make the POST request
        try:
            json_string = ''
            tmp_json = ''
            data = {}
            with requests.post(
                "http://localhost:8003/v1/completions",
                headers=headers,
                json=payload,
                stream=True
                ) as response:
                    response.raise_for_status()
                    if not response:
                        break
                    for chunk in response.iter_content(chunk_size=1,decode_unicode=True):
                        if chunk:
                            json_string = f"{json_string}{chunk}"
                        try:
                            data = json.loads(json_string)
                            json_string = ''
                        except json.JSONDecodeError as ed:
                            fake = 1
                            data = False
                            continue
                        # Extract the assistant response
                        if 1 == 1:
                            fake = True
                        if "content" in data:
                            content = data.get("content", "")#.strip() 
                            assistant_response = content #cleaner.extract_and_clean(content)
                        elif "choices" in data and len(data["choices"]) > 0:
                            assistant_response = cleaner.extract_and_clean(data["choices"][0].get("text", "").strip())
                        elif data:
                            assistant_response = f"{data}"
                        sys.stdout.write(f"{assistant_response}")
                        # Force the terminal to render the text immediately
                        sys.stdout.flush()

                        # The context prompt is always returned with the assistant response appended
                        context = data.get('context')

                        # Replace literal \s
                    context = context.replace("\\", "")
                    print("\n\n")

            #print("--- Response ---")
            #print(context)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
        #except Exception as e:
        #    print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
