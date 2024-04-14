from django.db import models
from django.db.models import DateTimeField

class Email(models.Model):
    subject = models.CharField(max_length=255)
    sender = models.EmailField()
    body = models.TextField()
    classified_as = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def str(self):
        return self.subject