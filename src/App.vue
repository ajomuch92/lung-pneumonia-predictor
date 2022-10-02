<template>
  <div id="app">
    <h1>Predictor de Pulmonía por Radiografía</h1>
    <h2>Selecciona una image de tu radiografía</h2>
    <input type="file" accept="image/*" @change="onFileChange"/>
    <p>Resultado:</p>
    <ul v-if="Array.isArray(result)">
      <li v-for="(item, key) in result" :key="key">{{item}}</li>
    </ul>
    <p v-else>{{result}}</p>
  </div>
</template>

<script>
import Jimp from 'jimp';
import * as tf from '@tensorflow/tfjs';

export default {
  name: 'App',
  data: () => ({
    result: '',
    modelCNN: undefined,
    modelCNNAD: undefined,
    modelCNN2AD: undefined,
    modelDensoAD: undefined,
  }),
  mounted() {
    this.loadModel();
  },
  methods: {
    async loadModel() {
      this.modelCNN = await tf.loadLayersModel('cnn/model.json');
      this.modelCNNAD = await tf.loadLayersModel('cnn_ad/model.json');
      this.modelCNN2AD = await tf.loadLayersModel('cnn2_ad/model.json');
      this.modelDensoAD = await tf.loadLayersModel('denso_ad/model.json');
    },
    onFileChange(e) {
      const files = e.target.files || e.dataTransfer.files;
      const reader = new FileReader();
      this.result = 'Cargando...'
      reader.onload = (ev) => {
        const { result } = ev.target;
        Jimp.read(result).then((image) => {
          const result2 = image.resize(100, 100);
          const { bitmap } = result2;
          this.predict(bitmap.data);
        })
      }
      if (files) {
        reader.readAsArrayBuffer(files[0]);
      }
    },
    predict(rawData) {
      const arr = [];
      let arr100 = [];
      for (let i = 0; i < rawData.length; i += 4) {
        const pixel = rawData[i] / 255;
        arr100.push([pixel]);
        if (arr100.length === 100) {
          arr.push(arr100);
          arr100 = [];
        }
      }
      const tensor = tf.tensor4d([arr]);
      const resultCNN = this.modelCNN.predict(tensor).dataSync();
      const resultCNNAD = this.modelCNNAD.predict(tensor).dataSync();
      const resultCNN2AD = this.modelCNN2AD.predict(tensor).dataSync();
      const resultDensoAd = this.modelDensoAD.predict(tensor).dataSync();
      this.result = [`Predicción Modelo CNN: ${resultCNN[0] < 0.5 ? 'Sano' : 'Con Neumonía'}`,
        `Predicción modelo CNN AD: ${resultCNNAD[0] < 0.5 ? 'Sano' : 'Con Neumonía'}`,
        `Predicción modelo CNN AD DropOut: ${resultCNN2AD[0] < 0.5 ? 'Sano' : 'Con Neumonía'}`,
        `Predicción modelo Denso: ${resultDensoAd[0] < 0.5 ? 'Sano' : 'Con Neumonía'}`];
    },
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 100%;
  height: 99vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
</style>
