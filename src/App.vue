<template>
  <div id="app">
    <h1>Predictor de Pulmonía por Radiografía</h1>
    <h2>Selecciona una image de tu radiografía</h2>
    <input type="file" accept="image/*" @change="onFileChange"/>
    <p>Resultado: {{result}}</p>
  </div>
</template>

<script>

export default {
  name: 'App',
  data: () => ({
    result: '',
  }),
  methods: {
    onFileChange(e) {
      const files = e.target.files || e.dataTransfer.files;
      const reader = new FileReader();
      reader.onload = (ev) => {
        console.log(ev.target.result);
      }
      if (files) {
        reader.readAsDataURL(files[0]);
      }
    }
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
