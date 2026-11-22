from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

@traceable(name="Calendar Agent Test")
def test_agent():
    print("LangSmith tracing is working!")
    return "Test successful"

if __name__ == "__main__":
    result = test_agent()
    print(result)