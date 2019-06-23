import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import VueCropper from 'vue-cropper'
import axios from 'axios'


Vue.use(VueCropper)
Vue.config.productionTip = false
Vue.use(ElementUI);
Vue.prototype.axios = axios


new Vue({
  router,
  store,
  el: '#app',
  render: h => h(App)
}).$mount('#app')
