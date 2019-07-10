<template>
  <div>
    <el-steps :active="active" simple style="margin-left:20%;margin-right:20%;margin-bottom:20px">
      <el-step title="步骤 1" icon="el-icon-edit" @click.native="active=1"></el-step>
      <el-step title="步骤 2" icon="el-icon-upload" @click.native="active=2"></el-step>
      <el-step title="步骤 3" icon="el-icon-picture" @click.native="active=3"></el-step>
    </el-steps>
    <div class="imp_processor" v-if="active==1">
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
          <!--<div class="el-upload__tip">支持绝大多数图片格式，单张图片最大支持5MB</div>-->
        </el-upload>

        <el-button type="text" @click="dialogVisible=true">重新裁剪</el-button>

        <div>
          <img :src="imgbase64" style="padding: 10px" />
          <br />
          <el-input
            type="textarea"
            autosize
            v-model="ocrText"
            placeholder="OCR识别结果在此"
            style="width: 60%"
            :disabled="true"
          ></el-input>
          <br />
          <div style="padding: 25px">
            <el-button
              style="margin-right: 25px"
              type="primary"
              plain
              :loading="buttonDisable"
              @click="ocr"
            >OCR识别</el-button>
            <el-button
              style="margin-left: 25px"
              type="success"
              plain
              :loading="buttonDisable"
              @click="nextstep"
            >下一步</el-button>
          </div>
        </div>

        <el-dialog
          title="图片剪裁"
          :visible.sync="dialogVisible"
          append-to-body
          :close-on-click-modal="false"
        >
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
                  <img :src="previews.url" :style="previews.img" />
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
    <div class="selector" v-if="active==2">
      <el-container>
        <el-header height="0px"></el-header>
        <el-main style="height: 300px; position: relative">
          <div style="position:absolute; text-align:center; margin:0 auto">
            <img :src="imgbase64" style />
            <el-tag
              size="mini"
              v-for="(tag, index) in ocrResult"
              :key="index"
              :style="{opacity:transparent / 100, position:`absolute`, left:tag.frame[0].split(`,`)[0] + `px`, top:tag.frame[0].split(`,`)[1] + `px`}"
              @click="addElement(tag)"
            >{{tag.content}}</el-tag>
          </div>
        </el-main>
        <el-main style="height: 300px">
          <el-table :data="candidateSegment" border style="width: 100%" v-loading="loading">
            <el-table-column label="正文" width>
              <template slot-scope="scope">
                <el-input v-model="scope.row.content" placeholder="请输入内容"></el-input>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="120">
              <template slot-scope="scope">
                <el-button
                  @click.native.prevent="deleteRow(scope.$index, candidateSegment)"
                  type="text"
                  size="small"
                >移除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-main>
        <el-footer>
          <el-button
            style="margin-left: 10px; margin-right: 10px"
            type="success"
            plain
            :loading="buttonDisable"
            @click="laststep"
          >上一步</el-button>
          <el-button
            style="margin-left: 10px; margin-right: 10px"
            type="success"
            plain
            :loading="buttonDisable"
            @click="nextstep"
          >下一步</el-button>
          <br>
          <el-button
            style="margin-top: 20px"
            type="danger"
            plain
            :loading="buttonDisable"
            @click="clearCandidateList"
          >清空</el-button>
          <el-row>
            <el-col :span="12" :offset="6">
              <div class="text">标签透明度</div>
              <el-slider v-model="transparent"></el-slider>
            </el-col>
          </el-row>
        </el-footer>
      </el-container>
    </div>
    <div class="segmentation" v-if="active==3">
      <el-button style type="success" plain :loading="buttonDisable" @click="segment">分析</el-button>

      <div v-for="(data, index) in segmentResult.datas" :key="index" style="margin-top:20px">
        <el-card class="box-card" style="width:fit-content; margin:0 auto">
          <div slot="header" class="clearfix">
            <span>{{segmentResult.origindatas[index]}}</span>
          </div>
          <div v-for="(value, key, i) in data" :key="i" style="margin-bottom:10px">
            <el-row type="flex" justify="center">
              <el-col :span="12">
                <el-badge is-dot class="item" :type="value==`Unlabled`?`error`:`success`">
                  <el-button size="mini">{{key}}</el-button>
                </el-badge>
              </el-col>
              <el-col :span="12" style="margin: auto; padding-left: 10px">{{value}}</el-col>
            </el-row>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script>
import { VueCropper } from 'vue-cropper'

export default {
  name: 'OrderReader',
  data() {
    return {
      serverURL: '/api',
      dialogVisible: false,
      option: {
        img: '', // 裁剪图片的地址
        info: true, // 裁剪框的大小信息
        outputSize: 1, // 裁剪生成图片的质量
        outputType: 'png', // 裁剪生成图片的格式
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
        centerBox: true, // 截图框是否被限制在图片里面
        infoTrue: true, // true 为展示真实输出图片宽高 false 展示看到的截图框宽高
      },
      previews: {},
      fileinfo: {},
      // 防止重复提交
      loading: false,
      imgbase64: '',
      ocrText: '',
      buttonDisable: false,
      active: 1,
      ocrResult: [],
      candidateSegment: [],
      loading: false,
      transparent: 80,
      segmentResult: [],
    }
  },
  methods: {
    changeUpload(file, fileList) {
      this.fileinfo = file
      // 上传成功后将图片地址赋值给裁剪框显示图片
      this.$nextTick(() => {
        console.log(file.raw)
        this.option.img = URL.createObjectURL(new Blob([file.raw]))
        this.dialogVisible = true
      })
    },
    finish() {
      var img
      this.$refs.cropper.getCropData(data => {
        try {
          img = data
        } catch (error) {
          img = window.URL.createObjectURL(data)
        }
        this.imgbase64 = img
        this.dialogVisible = false
        this.ocr()
      })
    },
    realTime(data) {
      this.previews = data
    },
    ocr() {
      this.buttonDisable = true
      this.loading = true
      this.axios
        .post(this.serverURL + '/ocr', {
          img: this.imgbase64,
        })
        .then(
          res => {
            this.ocrResult = res.data.result
            if (res.data.success == 0) {
              this.$message.error('OCR无法识别目标图片')
              this.ocrText = ''
              this.buttonDisable = false
              this.loading = false
              return
            }
            this.candidateSegment = JSON.parse(JSON.stringify(this.ocrResult))
            console.log(this.candidateSegment)
            this.buttonDisable = false
            this.loading = false

            var content = ''
            for (let index = 0; index < res.data.result.length; index++) {
              const element = res.data.result[index]
              content += element.content
            }
            this.ocrText = content
          },
          res => {
            this.buttonDisable = false
            this.loading = false
            this.$message.error('OCR后台服务错误')
          }
        )
    },

    segment() {
      var data = []
      var index = 0
      for (let i = 0; i < this.candidateSegment.length; i++) {
        const element = this.candidateSegment[i].content
        data[index++] = element
      }

      this.loading = true
      this.axios
        .post(this.serverURL + '/segment', {
          data: data,
        })
        .then(
          res => {
            this.loading = false
            this.segmentResult = res.data
            if (this.segmentResult.docs != '') {
              this.$message.error(this.segmentResult.docs)
            }
          },
          res => {
            this.loading = false
            this.$message.error('后台服务错误')
          }
        )
    },

    nextstep() {
      if (this.active < 3) {
        this.active = this.active + 1
      }
    },
    laststep() {
      if (this.active > 0) {
        this.active = this.active - 1
      }
    },
    addElement(element) {
      this.candidateSegment.push(element)
    },
    clearCandidateList() {
      console.log(this.candidateSegment)
      this.candidateSegment = []
    },
    deleteRow(index, rows) {
      rows.splice(index, 1)
      //console.log(this.ocrResult);
    },
  },

  computed: {
    previewStyle() {
      var previews = this.previews
      var cropperHeight = this.cropperHeight
      return {
        width: previews.w + 'px',
        height: previews.h + 'px',
        overflow: 'hidden',
        margin: 'auto',
        zoom: this.isLandscape
          ? this.previewWidth / previews.w
          : cropperHeight / previews.h,
      }
    },
  },
  components: {
    VueCropper,
  },
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.cropper {
  width: auto;
  height: 1200px;
}
.text {
  margin-top: 10px;
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB',
    'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
}
</style>
