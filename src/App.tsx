import React, { useRef, useState, useEffect } from "react";
import * as faceapi from "face-api.js";
import Webcam from "react-webcam";

type CharacterData = {
  name: string;
  path: string;
  descriptor: Float32Array;
};

const App = () => {
  const [image, setImage] = useState<string | null>(null);
  const [similarCharacter, setSimilarCharacter] = useState<CharacterData | null>(null);
  const [noMatchFound, setNoMatchFound] = useState<boolean>(false); // 유사한 이미지가 없을 때 표시
  const [modelsLoaded, setModelsLoaded] = useState<boolean>(false); // 모델 로딩 상태
  const [isComparing, setIsComparing] = useState<boolean>(false); // 비교 중 상태
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [characterData, setCharacterData] = useState<CharacterData[]>([]);

  // 캐릭터 이미지에서 임베딩 데이터 추출
  const loadCharacterEmbeddings = async () => {
    const characterImages = [
      { name: "TiniPing1", path: "/images/tiniping1.jpg" },
      { name: "TiniPing2", path: "/images/tiniping2.jpg" },
      { name: "TiniPing3", path: "/images/tiniping3.jpg" },
      { name: "TiniPing4", path: "/images/tiniping4.jpg" },
    ];

    const embeddings = await Promise.all(
      characterImages.map(async character => {
        const img = await faceapi.fetchImage(character.path);
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        if (detections) {
          return {
            name: character.name,
            path: character.path,
            descriptor: detections.descriptor,
          };
        }
        return null;
      }),
    );

    // 유효한 임베딩만 설정
    const validEmbeddings = embeddings.filter(embedding => embedding !== null) as CharacterData[];
    setCharacterData(validEmbeddings);
  };

  useEffect(() => {
    // 모델을 로드하고 임베딩 데이터를 준비
    const loadModels = async () => {
      try {
        const MODEL_URL = process.env.PUBLIC_URL + "/models";
        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        // await faceapi.nets.ssdMobilenetv1.loadFromUri("/models");
        // await faceapi.nets.faceRecognitionNet.loadFromUri("/models");
        // await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
        setModelsLoaded(true); // 모델이 로드되었음을 설정
        await loadCharacterEmbeddings();
      } catch (error) {
        console.error("모델 로딩 중 오류 발생:", error);
      }
    };

    loadModels();
  }, []);

  const capture = async () => {
    if (!modelsLoaded) {
      alert("모델이 아직 로드되지 않았습니다. 잠시만 기다려주세요.");
      return;
    }

    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setImage(imageSrc);
      setNoMatchFound(false); // 이전 결과 초기화
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result && typeof reader.result === "string") {
          setImage(reader.result);
          setNoMatchFound(false); // 이전 결과 초기화
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const compareFace = async () => {
    if (!image) {
      alert("비교할 이미지가 없습니다.");
      return;
    }

    if (!modelsLoaded) {
      alert("모델이 아직 로드되지 않았습니다. 잠시만 기다려주세요.");
      return;
    }

    setIsComparing(true); // 비교 시작
    await detectFace(image);
    setIsComparing(false); // 비교 완료
  };

  const detectFace = async (image: string) => {
    const img = await faceapi.fetchImage(image);
    const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

    if (detections) {
      const descriptor = detections.descriptor;
      const bestMatch = findBestMatch(descriptor);
      setSimilarCharacter(bestMatch);
    }
  };

  const findBestMatch = (descriptor: Float32Array) => {
    let minDistance = Infinity;
    let bestMatch: CharacterData | null = null;

    characterData.forEach(character => {
      const distance = faceapi.euclideanDistance(descriptor, character.descriptor);
      if (distance < minDistance) {
        minDistance = distance;
        bestMatch = character;
      }
    });

    // 임계값 설정 - 이 값을 넘어가면 "유사한 이미지 없음" 처리
    const threshold = 0.6;
    if (minDistance > threshold) {
      setNoMatchFound(true); // 유사한 이미지가 없는 경우
      return null;
    }

    setNoMatchFound(false); // 유사한 이미지가 있는 경우
    return bestMatch;
  };

  return (
    <div>
      <h1>TEST</h1>

      {/* 모델이 로드되기 전에는 아래 UI를 숨깁니다 */}
      {modelsLoaded ? (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
          }}
        >
          <Webcam audio={false} ref={webcamRef} screenshotFormat='image/jpeg' width={320} height={240} />
          <button onClick={capture} disabled={isComparing}>
            사진 찍기
          </button>

          <div style={{ marginTop: "20px" }}>
            <input type='file' accept='image/*' onChange={handleImageUpload} disabled={isComparing} />
          </div>
        </div>
      ) : (
        <p>모델을 로드 중입니다. 잠시만 기다려주세요...</p>
      )}

      {image && (
        <div>
          <img src={image} alt='Captured' style={{ marginTop: "20px" }} />
          <button onClick={compareFace} style={{ marginTop: "20px" }} disabled={isComparing}>
            {isComparing ? "비교 중..." : "비교하기"}
          </button>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: "none" }} />

      {isComparing && <p style={{ color: "blue" }}>비교 중입니다. 잠시만 기다려주세요...</p>}

      {noMatchFound ? (
        <div style={{ marginTop: "20px", color: "red" }}>
          <h2>유사한 이미지가 없습니다.</h2>
        </div>
      ) : similarCharacter ? (
        <div style={{ marginTop: "20px" }}>
          <h2>유사한 캐릭터</h2>
          <img src={similarCharacter.path} alt='Character' />
        </div>
      ) : null}
    </div>
  );
};

export default App;
