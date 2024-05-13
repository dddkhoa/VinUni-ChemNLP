### Instruction for running

```Docker

# For building the container

docker build -t chemnlp:v1.0 .

# For running

docker run -e "INPUT_FOLDER={PATH_TO_INPUT:/app/INPUT}" -e "OUTPUT_FOLDER={PATH_TO_OUTPUT:/app/OUTPUT}" -e "OPENAI_API_KEY={key}" {IMAGENAME:chemnlp}
```

### Outpur Directory Structure

```
OUTPUT
│
└───artifacts
│   │   parsed.csv # for all PDFs
│   │   table.csv  # for all PDFs

│
└───pdf1
│   │   pdf1.json
│   │
│   └───artifacts
│       │   pdf1.png  # for thumbnail

│
└───pdf2
│   │   pdf2.json
│   │
│   └───artifacts
│       │   pdf1.png  # for thumbnail
```
