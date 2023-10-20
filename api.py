from fastapi import FastAPI
from pydantic import BaseModel
from run_qa import get_qa


class InputData(BaseModel):
    message: str
    user_id: str


app = FastAPI()


@app.post("/message/")
def answer_question(input_data: InputData) -> str:
    input_question = input_data.message
    input_userid = input_data.user_id
    print('Userid:', input_userid)
    answer = get_qa(input_question)['result']
    print(f'=========INPUT: {input_question} ===========\n\n===========ANSWER: ===========\n{answer}')
    return answer


if __name__ == "__main__":
    import uvicorn
    print('[API] FastAPI is up. Please send your requests!')
    uvicorn.run(app, host="0.0.0.0", port=8080)
