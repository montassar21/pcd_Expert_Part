import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AlgorithmsService {
  apiUrl: string ="http://127.0.0.1:5000/";
  constructor(private httpClient: HttpClient) { }
  public addComment(loginObj:any){
    return this.httpClient.post<any>(`${this.apiUrl}addComment`,loginObj);
  }
}
