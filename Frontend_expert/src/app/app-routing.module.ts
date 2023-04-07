import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {UploadFileComponent} from './file-upload/file-upload.component';

const routes: Routes = [
  {path:'/upload' , component: UploadFileComponent}, 
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
