import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {UploadFileComponent} from './file-upload/file-upload.component';
import { AlgorithmsComponent } from './algorithms/algorithms.component';
import { AdminComponent } from './admin/admin.component';
import { CommentsAdminComponent } from './comments-admin/comments-admin.component';

const routes: Routes = [
  {path:'upload' , component: UploadFileComponent},
  {path:'algorithms' , component: AlgorithmsComponent},
  {path:'admin' , component: AdminComponent},
  {path:'commentsAdmin' , component:  CommentsAdminComponent
  },



];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
