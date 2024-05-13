from model import parse_pdfs, model_1, tabulate_condition

import json
import os


def extract_metadata(input_pdfs_path, pdf_titles, thumbnails, output_folder_path):
    assert len(input_pdfs_path) == len(pdf_titles) == len(thumbnails)

    for file, pdf_title, pdf_thumbnail in zip(input_pdfs_path, pdf_titles, thumbnails):
        # Extract filename without `.pdf` extension
        filename = os.path.basename(file)[:-4]
        pdf_thumbnail_path = os.path.join(
            output_folder_path, filename, "artifacts", f"{filename}.png"
        )
        pdf_thumbnail.save(pdf_thumbnail_path)

        metadata = {
            "title": pdf_title,
            "thumbnail": pdf_thumbnail_path,
            "parsed": os.path.join(output_folder_path, "artifacts", "parsed.csv"),
            "table": os.path.join(output_folder_path, "artifacts", "table.csv"),
        }

        with open(
            os.path.join(output_folder_path, filename, f"{filename}.json"), "w+"
        ) as f:
            json.dump(metadata, f)


if __name__ == "__main__":
    input_folder = os.environ.get("INPUT_FOLDER")
    output_folder = os.environ.get("OUTPUT_FOLDER")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    os.makedirs(os.path.join(output_folder, "artifacts"), exist_ok=True)

    input_pdfs_path = []
    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            input_pdfs_path.append(os.path.join(input_folder, file))
            filename = os.path.basename(file)[:-4]

            os.makedirs(os.path.join(output_folder, filename), exist_ok=True)

            os.makedirs(
                os.path.join(output_folder, filename, "artifacts"),
                exist_ok=True,
            )

    print("Processing dataframe")
    dataframe, pdf_titles, thumbnails = parse_pdfs(
        input_pdfs_path, filter_ref=True, combine=False
    )

    dataframe.to_csv(
        os.path.join(output_folder, "artifacts", "parsed.csv"), index=False
    )

    # TODO: logging with % task done

    # Dataframe contains conditions from all PDFs
    updated_dataframe = model_1(dataframe, openai_api_key)
    print("Processing table")
    # Table contains conditions from all PDFs

    table = tabulate_condition(updated_dataframe, "summarized")
    table.to_csv(os.path.join(output_folder, "artifacts", "table.csv"), index=False)

    print("Extracting metadata")
    extract_metadata(input_pdfs_path, pdf_titles, thumbnails, output_folder)
