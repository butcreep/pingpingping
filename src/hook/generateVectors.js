const tf = require("@tensorflow/tfjs-node");
const blazeface = require("@tensorflow-models/blazeface");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");

// 이미지 경로 설정
const tinipingImages = [
  { name: "TiniPing1", path: "./images/tiniping1.jpg" },
  { name: "TiniPing2", path: "./images/tiniping2.jpg" },
  { name: "TiniPing3", path: "./images/tiniping3.jpg" },
  { name: "TiniPing4", path: "./images/tiniping4.jpg" },
];

async function generateVectors() {
  const model = await blazeface.load();
  const vectors = [];

  for (const tiniping of tinipingImages) {
    const img = await loadImage(tiniping.path);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    const predictions = await model.estimateFaces(canvas, false);

    if (predictions.length > 0) {
      const vector = predictions[0].landmarks.flat(); // 특징 벡터 생성
      vectors.push({ name: tiniping.name, vector });
    } else {
      console.log(`No face detected in ${tiniping.name}`);
    }
  }

  // 벡터 데이터를 JSON 파일로 저장
  fs.writeFileSync("tinipingVectors.json", JSON.stringify(vectors, null, 2));
  console.log("벡터 데이터 생성 완료!");
}

generateVectors();
