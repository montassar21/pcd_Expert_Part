import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { FormGroup, FormControl, Validators, FormBuilder } from '@angular/forms';
import {FileUploadService} from './file-upload.service'
import { Router } from '@angular/router';
@Component({
  selector: 'app-upload-file',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css']
})
export class UploadFileComponent implements OnInit {
  [x: string]: any;

  uploadForm!: FormGroup;
  selectedFile!: File;
  csvFile!: File;
  accuracy!:string;
  LR!:any
  RF!:any
  GBM!:any
  SVM!:any
  DT!:any

  @ViewChild('fileInput') fileInput!: ElementRef;

  constructor(private fb:FormBuilder,
    private fileUploadService: FileUploadService,
    private router: Router) {
   }


ngOnInit(){

}


  fileToUpload: File | null = null;





  onFileSelected(event: any) {
    this.fileToUpload = event.target.files[0];

  }

  onUpload() {
    const formData = new FormData();
  if (this.fileToUpload) {
    formData.append('file', this.fileToUpload, this.fileToUpload.name);
    // envoyer le formData au serveur pour le traitement
  } else {
    console.log("No selected file !");
  }



    this.fileUploadService.uploadFile(formData).subscribe(
      (response) => {console.log(response);
       this.LR = response['LR'];
       this.RF = response['RF'];
       this.SVM = response['SVM'];
       this.GBM = response['GBM'];
       this.DT = response['DT'];

       console.log(response['message']);
       this.accuracy=response['accuracy']
        console.log(this.accuracy)

    },
      (error) => console.log(error)
    );
    ;
  }
  chooseAlgorithms(){

    this.router.navigate(['algorithms'],{queryParams:{data:this.accuracy,LR:this.LR,RF:this.RF,DT:this.DT,SVM:this.SVM,GBM:this.GBM}});
  }
}
