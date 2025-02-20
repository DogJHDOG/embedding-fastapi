from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchaudio
import numpy as np
from typing import Dict, List
import tempfile
import os
import uvicorn
from pydantic import BaseModel
import soundfile as sf

class AudioEmbeddingResponse(BaseModel):
    embedding: List[float]
    shape: List[int]

class AudioFeatureExtractor:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        """
        Args:
            model_path (str): TorchScript 모델 파일 경로 (.pt)
            sample_rate (int): 목표 샘플링 레이트
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        self.sample_rate = sample_rate

    async def process_audio(self, audio_file: UploadFile) -> np.ndarray:
        """
        업로드된 음성 파일을 처리하고 특징 벡터를 추출합니다.
        
        Args:
            audio_file (UploadFile): FastAPI UploadFile 객체
            
        Returns:
            np.ndarray: 추출된 특징 벡터
        """
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)
        
        try:
            # soundfile로 오디오 로드
            waveform, sr = sf.read(temp_path)
            
            # 스테레오를 모노로 변환 (필요한 경우)
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = np.mean(waveform, axis=1)
            
            # torch tensor로 변환 및 shape 조정
            waveform = torch.FloatTensor(waveform).unsqueeze(0)  # (1, T) 형태로 변환
            
            # 리샘플링 (필요한 경우)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 모델 입력을 위한 텐서 준비
            waveform = waveform.to(self.device)
            
            # 추론 모드로 특징 추출
            with torch.no_grad():
                features = self.model(waveform)
                
            # CPU로 이동 후 numpy 배열로 변환
            features = features.cpu().numpy()
            
            return features
            
        finally:
            # 임시 파일 삭제
            os.unlink(temp_path)

# FastAPI 앱 초기화
app = FastAPI(
    title="Audio Embedding API",
    description="음성 파일을 받아 임베딩 벡터를 반환하는 API",
    version="1.0.0"
)

# 전역 변수로 특징 추출기 초기화
MODEL_PATH = "model.pt"  # 실제 모델 경로로 수정 필요
extractor = AudioFeatureExtractor(MODEL_PATH)

@app.post("/extract_embedding", response_model=AudioEmbeddingResponse)
async def extract_embedding(audio_file: UploadFile = File(...)) -> Dict:
    """
    .wav 음성 파일을 받아서 임베딩 벡터를 추출합니다.
    
    Args:
        audio_file (UploadFile): .wav 형식의 음성 파일
        
    Returns:
        Dict: 임베딩 벡터와 shape 정보를 포함하는 딕셔너리
    """
    # 파일 형식 검증
    if not audio_file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="WAV 형식의 파일만 지원됩니다."
        )
    
    try:
        # 특징 추출
        features = await extractor.process_audio(audio_file)
        
        # 결과 반환
        return {
            "embedding": features.flatten().tolist(),
            "shape": list(features.shape)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 추출 중 오류 발생: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)