import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http'; // Import HttpClientModule
import { FormsModule, ReactiveFormsModule } from '@angular/forms'; // Import ReactiveFormsModule
import { AppComponent } from './app.component';
import { UploadFileComponent } from './file-upload/file-upload.component';
import { AlgorithmsComponent } from './algorithms/algorithms.component';
import { AppRoutingModule } from './app-routing.module';
import { NgToastModule } from 'ng-angular-popup';
import { AdminComponent } from './admin/admin.component';
import { CommentsAdminComponent } from './comments-admin/comments-admin.component';

@NgModule({
  declarations: [
    AppComponent,
    UploadFileComponent,
    AlgorithmsComponent,
    AdminComponent,
    CommentsAdminComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    NgToastModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
