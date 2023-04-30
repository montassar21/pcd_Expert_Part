import { Component, Input, OnInit,AfterViewInit } from '@angular/core';
import { AlgorithmsService } from './algorithms.service';
import { ActivatedRoute, Router } from '@angular/router';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { FileUploadService } from '../file-upload/file-upload.service';
import {Location, getLocaleCurrencyCode} from '@angular/common'
import { HttpClient } from '@angular/common/http';
import {NgToastService} from 'ng-angular-popup'
@Component({
  selector: 'app-algorithms',
  templateUrl: './algorithms.component.html',
  styleUrls: ['./algorithms.component.css']
})
export class AlgorithmsComponent implements OnInit {
  accuracy!:string
  cm:string='../../assets/noimg.png'
  curve:string='../../assets/noimg.png'
  acc:string='../../assets/noimg.png'
  report:string='../../assets/noimg.png'
 formDat!:FormGroup
  constructor(private fb:FormBuilder,
    private algoService: AlgorithmsService,
    private router: Router,private fileUploadService:FileUploadService,private location:Location,private route:ActivatedRoute,private http: HttpClient,private algoServices:AlgorithmsService
    ,private toast: NgToastService

    ){}
ngOnInit(): void {
    this.route.queryParams.subscribe((params:any)=>{
      console.log(params);

    })
    this.formDat=this.fb.group({

      name:'',
      comment:['',Validators.required],
      })


}

executeLR() {
  this.route.queryParams.subscribe((params:any)=>{
    console.log(params.data[0]);

    this.cm=`../../assets/${params.LR[0]}`;
    this.curve=`../../assets/${params.LR[1]}`;
    this.acc='../../assets/accuracyLR.png'
    this.accuracy=params.data[0]
    this.report='../../assets/c_reportLR.png'

  })
}
executeRF() {
  this.route.queryParams.subscribe((params:any)=>{
    console.log(params.data[1]);

    this.cm=`../../assets/${params.RF[0]}`;
    this.curve=`../../assets/${params.RF[1]}`;
    this.acc='../../assets/accuracyRF.png'
    this.accuracy=params.data[1]
    this.report='../../assets/c_reportRF.png'


  })
}
executeDT() {
  this.route.queryParams.subscribe((params:any)=>{
    console.log(params.data[2]);

    this.cm=`../../assets/${params.DT[0]}`;
    this.curve=`../../assets/${params.DT[1]}`;
    this.acc='../../assets/accuracyDT.png'
    this.accuracy=params.data[2]
    this.report='../../assets/c_reportDT.png'




})
}
executeGBC() {
  this.route.queryParams.subscribe((params:any)=>{
    console.log(params.data[3]);

    this.cm=`../../assets/${params.GBM[0]}`;
    this.curve=`../../assets/${params.GBM[1]}`;
    this.acc='../../assets/accuracyGBM.png'
    this.accuracy=params.data[3]
    this.report='../../assets/c_reportGBM.png'

  })
}
executeSVM() {
  this.route.queryParams.subscribe((params:any)=>{
    console.log(params.data[4]);

    this.cm=`../../assets/${params.SVM[0]}`;
    this.curve=`../../assets/${params.SVM[1]}`;
    this.acc='../../assets/accuracySVM.png'
    this.accuracy=params.data[4]
    this.report='../../assets/c_reportSVM.png'
  })
}
addComment(a:any){
  this.formDat.value.name=a
  this.algoService.addComment(this.formDat.value).subscribe({
    next:(res=>{
      if(res.Message=="Success !"){
      this.toast.success({detail:"SUCCESS",summary:res.Message,duration:5000});
    this.formDat.reset()}
    }),
    error:(err=>{
      this.toast.error({detail:"ERROR",summary:"Something went wrong !",duration:5000});
    })
  })
}

}
