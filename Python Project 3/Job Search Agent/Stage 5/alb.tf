# Application Load Balancer — the piece App Runner couldn't be:
# WebSocket-capable (Streamlit's UI runs entirely over one) with a long idle
# timeout so a multi-minute pipeline run doesn't drop the socket.
#
# HTTP-only by design: HTTPS on an ALB requires an ACM certificate, which
# requires a domain you own. The README documents this honestly (demo scope;
# Route 53 + ACM is the natural extension).

resource "aws_lb" "app" {
  name               = "${var.app_name}-alb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  # Streamlit keeps a long-lived WebSocket open; default 60s would sever it
  # mid-pipeline-run.
  idle_timeout = 3600
}

resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg"
  port        = 8501
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = data.aws_vpc.default.id

  health_check {
    path                = "/_stcore/health"
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }

  # Streamlit sessions are in-memory per instance; harmless at desired_count
  # 1, correct if ever scaled.
  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
