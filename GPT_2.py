import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 불러오기
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 모델을 평가 모드로 설정
model.eval()

# 텍스트 생성 함수
def generate_text(prompt, max_length=1000, num_return_sequences=1, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # GPU가 사용 가능한 경우 GPU로 모델을 이동
    if torch.cuda.is_available():
        model.to('cuda')
        input_ids = input_ids.to('cuda')

    # 텍스트 생성
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=max_length + len(input_ids[0]), # 입력 길이 포함하여 최대 길이 설정
            num_return_sequences=num_return_sequences, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,           # 샘플링 사용
            top_k=top_k,              # top_k 샘플링
            top_p=top_p               # top_p 샘플링
        )
    
    # 결과 디코딩 및 출력
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    
    # 프롬프트 부분을 제거하고 답변만 반환
    generated_responses = [text[len(prompt):].strip() for text in generated_texts]
    return generated_responses

# 예제 사용
prompt ="What is president's name?"
generated_text = generate_text(prompt, max_length=100, num_return_sequences=1)
print("프롬프트 :", prompt)
print("===================================================")
print("생성 결과:", generated_text[0])