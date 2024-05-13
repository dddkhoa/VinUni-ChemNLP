import fitz
import openai
import pandas as pd
import tiktoken


import re
import time


def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def parse_pdfs(pdf_files, filter_ref=True, combine=False):
    """Convert pdf files to dataframe and extract titles"""
    # Create an empty list to store the data
    data = []
    pdf_titles = []
    thumbnails = []
    # Iterate over the PDF files
    for pdf in pdf_files:
        print("Parsing PDF:", pdf)
        # Open the PDF file
        pdf_document = fitz.open(pdf)
        # Get the PDF title
        if pdf_document.metadata.get("title") == "":
            pdf_title = pdf.split("/")[-1]
        else:
            pdf_title = pdf_document.metadata.get("title")
        pdf_titles.append(pdf_title)

        # Get PDF thumbnail
        first_page = pdf_document.load_page(0)

        # Render the first page as an image
        pixmap = first_page.get_pixmap()
        thumbnails.append(pixmap)

        # Iterate over all the pages in the PDF
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            words = page_text.split()  # Split the page text into individual words
            page_text_join = " ".join(
                words
            )  # Join the words back together with a single space between each word

            if filter_ref:  # Filter the reference at the end
                page_text_join = remove_ref(page_text_join)

            page_len = len(page_text_join)
            div_len = page_len // 4  # Divide the page into 4 parts
            page_parts = [
                page_text_join[i * div_len : (i + 1) * div_len] for i in range(4)
            ]

            min_tokens = 40
            for i, page_part in enumerate(page_parts):
                if count_tokens(page_part) > min_tokens:
                    # Append the data to the list
                    data.append(
                        {
                            "file name": pdf,
                            "page number": page_num + 1,
                            "page section": i + 1,
                            "content": page_part,
                            "tokens": count_tokens(page_part),
                        }
                    )
        # Close the PDF document
        pdf_document.close()

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    if combine:
        df = combine_section(df)
    return df, pdf_titles, thumbnails


def remove_ref(pdf_text):
    """This function removes reference section from a given PDF text. It uses regular expressions to find the index of the words to be filtered out."""
    # Regular expression pattern for the words to be filtered out
    pattern = r"(REFERENCES|Acknowledgment|ACKNOWLEDGMENT)"
    match = re.search(pattern, pdf_text)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        clean_text = pdf_text[:start_index].strip()
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            "\[[\d\w]{1,3}\].+?[\d]{3,5}\.",
            "\[[\d\w]{1,3}\].+?[\d]{3,5};",
            "\([\d\w]{1,3}\).+?[\d]{3,5}\.",
            "\[[\d\w]{1,3}\].+?[\d]{3,5},",
            "\([\d\w]{1,3}\).+?[\d]{3,5},",
            "\[[\d\w]{1,3}\].+?[\d]{3,5}",
            "[\d\w]{1,3}\).+?[\d]{3,5}\.",
            "[\d\w]{1,3}\).+?[\d]{3,5}",
            "\([\d\w]{1,3}\).+?[\d]{3,5}",
            "^[\w\d,\.– ;)-]+$",
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = (
                re.sub(pattern, "", pdf_text)
                if len(matches) > 500
                and matches.count(".") < 2
                and matches.count(",") < 2
                and not matches[-1].isdigit()
                else pdf_text
            )

        # Split the text into lines
        lines = pdf_text.split("\n")

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = (
                    re.sub(pattern, "", lines[i])
                    if len(matches) > 500
                    and len(re.findall("\d", matches)) < 8
                    and len(set(matches)) > 10
                    and matches.count(",") < 2
                    and len(matches) > 20
                    else lines[i]
                )

        # Join the lines back together, excluding any empty lines
        clean_text = "\n".join([line for line in lines if line])

    return clean_text


def combine_section(df):
    """Merge sections, page numbers, add up content, and tokens based on the pdf name."""
    aggregated_df = (
        df.groupby("file name")
        .agg({"content": aggregate_content, "tokens": aggregate_tokens})
        .reset_index()
    )

    return aggregated_df


def aggregate_content(series):
    """Join all elements in the series with a space separator."""
    return " ".join(series)


def aggregate_tokens(series):
    """Sum all elements in the series."""
    return series.sum()


def extract_title(file_name):
    """Extract the main part of the file name."""
    title = file_name.split("_")[0]
    return title.rstrip(".pdf")


def combine_main_SI(df):
    """Create a new column with the main part of the file name, group the DataFrame by the new column,
    and aggregate the content and tokens."""
    df["main_part"] = df["file name"].apply(extract_title)
    merged_df = (
        df.groupby("main_part").agg({"content": "".join, "tokens": sum}).reset_index()
    )

    return merged_df.rename(columns={"main_part": "file name"})


def df_to_csv(df, file_name):
    """Write a DataFrame to a CSV file."""
    df.to_csv(file_name, index=False, escapechar="\\")


def csv_to_df(file_name):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_name)


def tabulate_condition(df, column_name):
    """This function converts the text from a ChatGPT conversation into a DataFrame.
    It also cleans the DataFrame by dropping additional headers and empty lines."""

    table_text = df[column_name].str.cat(sep="\n")

    # Remove leading and trailing whitespace
    table_text = table_text.strip()

    # Split the table into rows
    rows = table_text.split("\n")

    # Extract the header row and the divider row
    header_row, divider_row, *data_rows = rows

    # Extract column names from the header row

    column_names = [
        "paper name",
        "compound name",
        "metal source",
        "metal amount",
        "linker",
        "linker amount",
        "modulator",
        "modulator amount or volume",
        "solvent",
        "solvent volume",
        "reaction temperature",
        "reaction time",
    ]

    # Create a list of dictionaries to store the table data
    data = []

    # Process each data row
    for row in data_rows:
        # Split the row into columns
        columns = [col.strip() for col in row.split("|") if col.strip()]

        # Create a dictionary to store the row data
        row_data = {
            col_name: col_value for col_name, col_value in zip(column_names, columns)
        }

        # Append the dictionary to the data list
        data.append(row_data)

    df = pd.DataFrame(data)

    """Make df clean by drop additional header and empty lines """

    def contains_pattern(s, patterns):
        return any(re.search(p, s) for p in patterns)

    def drop_rows_with_patterns(df, column_name):
        # empty cells, N/A cells and header cells
        patterns = [
            r"^\s*$",
            r"--",
            r"-\s-",
            r"compound",
            r"Compound",
            r"Compound name",
            r"Compound Name",
            r"NaN",
            r"N/A",
            r"n/a",
            r"\nN/A",
            r"note",
            r"Note",
        ]

        mask = df[column_name].apply(lambda x: not contains_pattern(str(x), patterns))
        filtered_df = df[mask]

        return filtered_df

    # drop the repeated header
    df = drop_rows_with_patterns(df, "compound name")

    # drop the organic synthesis (where the metal source is N/a)
    filtered_df = drop_rows_with_patterns(
        drop_rows_with_patterns(
            drop_rows_with_patterns(df, "metal source"), "metal amount"
        ),
        "linker amount",
    )

    # drop the N/A rows
    filtered_df = filtered_df.dropna(
        subset=["metal source", "metal amount", "linker amount"]
    )

    return filtered_df


def split_content(input_string, tokens):
    """Splits a string into chunks based on a maximum token count."""

    MAX_TOKENS = tokens
    split_strings = []
    current_string = ""
    tokens_so_far = 0

    for word in input_string.split():
        # Check if adding the next word would exceed the max token limit
        if tokens_so_far + count_tokens(word) > MAX_TOKENS:
            # If we've reached the max tokens, look for the last dot or newline in the current string
            last_dot = current_string.rfind(".")
            last_newline = current_string.rfind("\n")

            # Find the index to cut the current string
            cut_index = max(last_dot, last_newline)

            # If there's no dot or newline, we'll just cut at the max tokens
            if cut_index == -1:
                cut_index = MAX_TOKENS

            # Add the substring to the result list and reset the current string and tokens_so_far
            split_strings.append(current_string[: cut_index + 1].strip())
            current_string = current_string[cut_index + 1 :].strip()
            tokens_so_far = count_tokens(current_string)

        # Add the current word to the current string and update the token count
        current_string += " " + word
        tokens_so_far += count_tokens(word)

    # Add the remaining current string to the result list
    split_strings.append(current_string.strip())

    return split_strings


def table_text_clean(text):
    """Cleans the table string and splits it into lines."""

    # Pattern to find table starts
    pattern = r"\|\s*compound\s*.*"

    # Use re.finditer() to find all instances of the pattern in the string and their starting indexes
    matches = [
        match.start() for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]

    # Count the number of matches
    num_matches = len(matches)

    # Base table string
    table_string = """|paper name | compound name | metal source | metal amount | linker | linker amount | modulator | modulator amount or volume | solvent | solvent volume | reaction temperature | reaction time |\n|---------------|-------|--------------|--------|---------------|-----------|---------------------------|---------|----------------|---------------------|---------------|\n"""

    if num_matches == 0:  # No table in the answer
        print("No table found in the text: " + text)
        splited_text = ""

    else:  # Split the text based on header
        splited_text = ""
        for i in range(num_matches):
            # Get the relevant table slice
            splited = (
                text[matches[i] : matches[i + 1]]
                if i != (num_matches - 1)
                else text[matches[i] :]
            )

            # Remove the text after last '|'
            last_pipe_index = splited.rfind("|")
            splited = splited[: last_pipe_index + 1]

            # Remove the header and \------\
            pattern_dash = r"-(\s*)\|"
            match = max(
                re.finditer(pattern_dash, splited),
                default=None,
                key=lambda x: x.start(),
            )

            if not match:
                print("'-|' pattern not found.")
            else:
                first_pipe_index = match.start()
                splited = (
                    "\n" + splited[(first_pipe_index + len("-|\n|") - 1) :]
                )  # Start from "\"

            splited_text += splited

    table_string = table_string + splited_text
    return table_string


def model_1(df, api_key):
    client = openai.Client(api_key=api_key)

    """Model 1 will turn text in dataframe to a summarized reaction condition table.The dataframe should have a column "file name" and a column "exp content"."""
    response_msgs = []

    for index, row in df.iterrows():
        print(f"Processing: {index+1}/{len(df)}")

        column1_value = row[df.columns[0]]
        column2_value = row["content"]

        max_tokens = 3000
        if count_tokens(column2_value) > max_tokens:
            context_list = split_content(column2_value, max_tokens)
        else:
            context_list = [column2_value]

        answers = ""  # Collect answers from chatGPT
        for idx, context in enumerate(context_list):
            user_heading = f"This is an experimental section on MOF synthesis from paper {column1_value}\n\nContext:\n{context}"
            user_ending = """Q: Can you summarize the following details in a table: 
            compound name or chemical formula (if the name is not provided), metal source, metal amount, organic linker(s), 
            linker amount, modulator, modulator amount or volume, solvent(s), solvent volume(s), reaction temperature, 
            and reaction time? If any information is not provided or you are unsure, use "N/A". 
            Please focus on extracting experimental conditions from only the MOF synthesis and ignore information related to organic linker synthesis, 
            MOF postsynthetic modification, high throughput (HT) experiment details or catalysis reactions. 
            If multiple conditions are provided for the same compound, use multiple rows to represent them. If multiple units or components are provided for the same factor (e.g.  g and mol for the weight, multiple linker or metals, multiple temperature and reaction time, mixed solvents, etc), include them in the same cell and separate by comma.
            The table should have 12 columns, all in lowercase:
            |paper name | compound name | metal source | metal amount | linker | linker amount | modulator | modulator amount or volume | solvent | solvent volume | reaction temperature | reaction time |

            A:"""

            attempts = 3
            while attempts > 0:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-0125",
                        messages=[
                            {
                                "role": "system",
                                "content": """Answer the question as truthfully as possible using the provided context,
                                        and if the answer is not contained within the text below, say "N/A" """,
                            },
                            {"role": "user", "content": user_heading + user_ending},
                        ],
                    )
                    answer_str = response.choices[0].message.content
                    if not answer_str.lower().startswith("n/a"):
                        answers += "\n" + answer_str
                    break
                except Exception as e:
                    attempts -= 1
                    if attempts <= 0:
                        print(
                            f"Error: Failed to process paper {column1_value}. Skipping. (model 1)"
                        )
                        break
                    print(
                        f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)"
                    )
                    time.sleep(60)

        response_msgs.append(answers)
    df = df.copy()
    df.loc[:, "summarized"] = response_msgs
    return df
