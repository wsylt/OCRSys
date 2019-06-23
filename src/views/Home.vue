<!--<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <HelloWorld msg="Welcome to Your Vue.js App"/>
  </div>
</template>

<script>
// @ is an alias to /src
import HelloWorld from '@/components/HelloWorld.vue'

export default {
  name: 'home',
  components: {
    HelloWorld
  }
}
</script>
-->

<!--
<template>
  <el-upload
    class="upload-demo"
    drag
    action="https://jsonplaceholder.typicode.com/posts/"
    multiple>
    <i class="el-icon-upload"></i>
    <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
    <div class="el-upload__tip" slot="tip">只能上传jpg/png文件，且不超过500kb</div>
  </el-upload>
    
</template>

<script>

</script>
-->

<template>
  <div>
    <div class="imp_processor">
      <div>
        <el-upload
          class="upload-demo"
          action
          drag
          :auto-upload="false"
          :show-file-list="false"
          :on-change="changeUpload"
        >
          <i class="el-icon-upload"></i>
          <div class="el-upload__text">拖拽 或 点击上传</div>
          <div class="el-upload__tip">支持绝大多数图片格式，单张图片最大支持5MB</div>
        </el-upload>

        <div>
          <img :src="imgbase64" style="padding: 10px">
          <br>
          <el-input type="textarea" v-model="ocrResult" placeholder="OCR识别结果在此" style="width: 30%"></el-input>
          <br>
          <div style="padding: 25px">
            <el-button style="margin-right: 25px" type="primary" plain :loading="buttonDisable" @click="ocr">OCR识别</el-button>
            <el-button style="margin-left: 25px" type="success" plain :loading="buttonDisable" >下一步</el-button>
          </div>
        </div>

        <el-dialog title="图片剪裁" :visible.sync="dialogVisible" append-to-body>
          <div class="cropper-content">
            <div class="cropper" style="text-align:center; height: 300px">
              <vueCropper
                ref="cropper"
                :img="option.img"
                :outputSize="option.size"
                :outputType="option.outputType"
                :info="true"
                :full="option.full"
                :canMove="option.canMove"
                :canMoveBox="option.canMoveBox"
                :original="option.original"
                :autoCrop="option.autoCrop"
                :autoCropWidth="option.autoCropWidth"
                :autoCropHeight="option.autoCropHeight"
                :fixedBox="option.fixedBox"
                :centerBox="option.centerBox"
                @realTime="realTime"
              ></vueCropper>
            </div>
            <div :style="{'margin-left':'20px'}">
              <div class="show-preview" :style="previewStyle">
                <div :style="previews.div" class="preview">
                  <img :src="previews.url" :style="previews.img">
                </div>
              </div>
            </div>
          </div>
          <div slot="footer" class="dialog-footer">
            <el-button @click="dialogVisible = false">取消</el-button>
            <el-button type="primary" @click="finish" :loading="loading">确认</el-button>
          </div>
        </el-dialog>
      </div>
    </div>
  </div>
</template>

<script>
import { VueCropper } from "vue-cropper";

export default {
  name: "OrderReader",
  data() {
    return {
      serverURL: "/api",
      dialogVisible: false,
      option: {
        img: "", // 裁剪图片的地址
        info: true, // 裁剪框的大小信息
        outputSize: 1, // 裁剪生成图片的质量
        outputType: "png", // 裁剪生成图片的格式
        canScale: true, // 图片是否允许滚轮缩放
        autoCrop: true, // 是否默认生成截图框
        autoCropWidth: 300, // 默认生成截图框宽度
        autoCropHeight: 100, // 默认生成截图框高度
        fixedBox: false, // 固定截图框大小 不允许改变
        fixed: false, // 是否开启截图框宽高固定比例
        fixedNumber: [5, 1], // 截图框的宽高比例
        full: true, // 是否输出原图比例的截图
        canMoveBox: true, // 截图框能否拖动
        original: false, // 上传图片按照原始比例渲染
        centerBox: false, // 截图框是否被限制在图片里面
        infoTrue: true // true 为展示真实输出图片宽高 false 展示看到的截图框宽高
      },
      previews: {},
      fileinfo: {},
      // 防止重复提交
      loading: false,
      imgbase64: "",
      ocrResult: "",
      buttonDisable: false,
    };
  },
  methods: {
    changeUpload(file, fileList) {
      //console.log(file);
      this.fileinfo = file;
      // 上传成功后将图片地址赋值给裁剪框显示图片
      this.$nextTick(() => {
        console.log(file.raw);
        this.option.img = URL.createObjectURL(new Blob([file.raw]));
        this.dialogVisible = true;
      });
    },
    finish() {
      //console.log(this.previews);
      var img;
      this.$refs.cropper.getCropData(data => {
        try {
          img = data;
        } catch (error) {
          img = window.URL.createObjectURL(data);
        }
        //let img = window.URL.createObjectURL(data);
        this.imgbase64 = img;
        this.dialogVisible = false;
        //console.log(this.imgbase64);
      });
    },
    realTime(data) {
      //console.log("realTime");
      //console.log(data)
      this.previews = data;
    },
    ocr() {
      this.buttonDisable = true;
      this.axios
        .post(this.serverURL, {
          img: this.imgbase64
        })
        .then(
          res => {
            console.log(res.data);
            this.ocrResult = res.data;
            this.buttonDisable = false;
          },
          res => {
            // 错误回调
          }
        );
    }
  },
  computed: {
    previewStyle() {
      var previews = this.previews;
      var cropperHeight = this.cropperHeight;
      return {
        width: previews.w + "px",
        height: previews.h + "px",
        overflow: "hidden",
        margin: "auto",
        zoom: this.isLandscape
          ? this.previewWidth / previews.w
          : cropperHeight / previews.h
      };
    }
  },
  components: {
    VueCropper
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.cropper {
  width: auto;
  height: 1200px;
}
</style>
