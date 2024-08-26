import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
const MainPing = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [detectionResult, setDetectionResult] = useState<string | null>(null);

  // 모델 로드
  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = process.env.PUBLIC_URL + "/models";
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      setModelsLoaded(true);
    };

    loadModels();
  }, []);

  // 웹캠 시작
  useEffect(() => {
    if (modelsLoaded && videoRef.current) {
      navigator.mediaDevices
        .getUserMedia({ video: {} })
        .then(stream => {
          videoRef.current!.srcObject = stream;
        })
        .catch(err => console.error("Error accessing webcam:", err));
    }
  }, [modelsLoaded]);

  const handleVideoPlay = async () => {
    if (videoRef.current) {
      const detections = await faceapi.detectSingleFace(videoRef.current!).withFaceLandmarks().withFaceDescriptor();

      if (detections) {
        const vector = Array.from(detections.descriptor); // 얼굴 특징 벡터 생성
        setDetectionResult(`Detected face with descriptor: ${vector.slice(0, 5).join(", ")}...`); // 간단히 첫 5개만 출력
      }
    }
  };

  return (
    <div className='App'>
      <h1>TiniPing Matcher</h1>
      {modelsLoaded ? (
        <div>
          <video ref={videoRef} onPlay={handleVideoPlay} autoPlay muted width='720' height='560' />
          {detectionResult && <p>{detectionResult}</p>}
        </div>
      ) : (
        <p>Loading models...</p>
      )}
    </div>
  );
};

export default MainPing;
