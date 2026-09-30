output "public_ip" {
  value = aws_instance.modelguard.public_ip
}

output "api_url" {
  value = "http://${aws_instance.modelguard.public_ip}:8000"
}

output "dashboard_url" {
  value = "http://${aws_instance.modelguard.public_ip}:8501"
}

output "ssh_command" {
  value = "ssh -i ${local_sensitive_file.ssh_key.filename} ubuntu@${aws_instance.modelguard.public_ip}"
}
