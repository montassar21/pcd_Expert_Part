import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CommentsAdminComponent } from './comments-admin.component';

describe('CommentsAdminComponent', () => {
  let component: CommentsAdminComponent;
  let fixture: ComponentFixture<CommentsAdminComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CommentsAdminComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CommentsAdminComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
