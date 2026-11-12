"""Start one local RAG API process; importing this file does not start a server."""
from pathlib import Path
from dotenv import load_dotenv


def main():
    load_dotenv(Path(__file__).with_name('.env'), encoding='utf-8-sig')
    import uvicorn
    from service_api import create_app
    app = create_app()
    uvicorn.run(app, host='127.0.0.1', port=8000, workers=1, access_log=False)


if __name__ == '__main__':
    main()
