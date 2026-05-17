

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime


SMTP_SETTINGS = {
    "gmail":   {"host": "smtp.gmail.com",   "port": 587},
    "outlook": {"host": "smtp.office365.com","port": 587},
    "yahoo":   {"host": "smtp.mail.yahoo.com","port": 587},
}


def _alert_html(location: str, vehicle_count: int,
                frame_id: int, threshold: int, severity: str) -> str:
    color = {"HIGH": "#f85149", "MODERATE": "#d29922", "LOW": "#2ea043"}[severity]
    icon  = {"HIGH": "🚨", "MODERATE": "⚠️", "LOW": "✅"}[severity]
    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:24px">
      <div style="max-width:520px;margin:auto;background:#161b22;
                  border:1px solid #30363d;border-radius:12px;padding:28px">
        <h2 style="color:{color};margin-top:0">{icon} {severity} CONGESTION ALERT</h2>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="padding:6px 0;color:#8b949e">📍 Location</td>
              <td style="padding:6px 0"><b>{location}</b></td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🚗 Vehicles Detected</td>
              <td style="padding:6px 0"><b style="color:{color}">{vehicle_count}</b></td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🎯 Threshold</td>
              <td style="padding:6px 0">{threshold}</td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🎞️ Frame</td>
              <td style="padding:6px 0">{frame_id}</td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🕐 Time</td>
              <td style="padding:6px 0">{datetime.now().strftime('%d %b %Y  %H:%M:%S')}</td></tr>
        </table>
        <div style="margin-top:18px;padding:12px;background:#21262d;
                    border-radius:8px;border-left:4px solid {color}">
          <b>Action Required:</b> {
            'Immediate rerouting recommended. Deploy traffic personnel.'
            if severity == 'HIGH' else
            'Monitor the situation. Consider signal adjustments.'
            if severity == 'MODERATE' else
            'Traffic is flowing normally.'
          }
        </div>
        <p style="color:#8b949e;font-size:12px;margin-top:20px">
          Sent by Traffic Congestion Detection System
        </p>
      </div>
    </body></html>
    """


def _summary_html(location: str, filename: str, total_frames: int,
                  max_vehicles: int, congested_frames: int,
                  congestion_rate: float, threshold: int) -> str:
    color = "#f85149" if congestion_rate > 50 else \
            "#d29922" if congestion_rate > 20 else "#2ea043"
    level = "HIGH" if congestion_rate > 50 else \
            "MODERATE" if congestion_rate > 20 else "LOW"
    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:24px">
      <div style="max-width:520px;margin:auto;background:#161b22;
                  border:1px solid #30363d;border-radius:12px;padding:28px">
        <h2 style="color:#58a6ff;margin-top:0">📊 Detection Complete — Summary Report</h2>
        <div style="background:{color}22;border:1px solid {color};border-radius:8px;
                    padding:12px 16px;margin-bottom:20px;text-align:center">
          <span style="font-size:1.4rem;font-weight:bold;color:{color}">
            {level} CONGESTION — {congestion_rate:.1f}%
          </span>
        </div>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="padding:6px 0;color:#8b949e">📍 Location</td>
              <td style="padding:6px 0"><b>{location}</b></td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">📁 File</td>
              <td style="padding:6px 0">{filename}</td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🎞️ Total Frames</td>
              <td style="padding:6px 0">{total_frames}</td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🚗 Max Vehicles</td>
              <td style="padding:6px 0"><b>{max_vehicles}</b></td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🔴 Congested Frames</td>
              <td style="padding:6px 0"><b style="color:{color}">{congested_frames}</b></td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">📈 Congestion Rate</td>
              <td style="padding:6px 0">
                <b style="color:{color}">{congestion_rate:.1f}%</b></td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🎯 Threshold Used</td>
              <td style="padding:6px 0">{threshold}</td></tr>
          <tr><td style="padding:6px 0;color:#8b949e">🕐 Generated</td>
              <td style="padding:6px 0">
                {datetime.now().strftime('%d %b %Y  %H:%M:%S')}</td></tr>
        </table>
        <p style="color:#8b949e;font-size:12px;margin-top:20px">
          Sent by Traffic Congestion Detection System
        </p>
      </div>
    </body></html>
    """


def send_email(sender_email: str, sender_password: str,
               recipient_email: str, subject: str,
               html_body: str, provider: str = "gmail") -> tuple[bool, str]:
    """
    Send an HTML email via STARTTLS.
    Returns (success: bool, message: str).
    """
    if provider not in SMTP_SETTINGS:
        return False, f"Unknown provider '{provider}'. Choose: {list(SMTP_SETTINGS)}"

    cfg = SMTP_SETTINGS[provider]
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender_email
        msg["To"]      = recipient_email
        msg.attach(MIMEText(html_body, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        return True, "Email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, (
            "Authentication failed. "
            "For Gmail: use an App Password, not your regular password. "
            "Generate one at https://myaccount.google.com/apppasswords"
        )
    except smtplib.SMTPConnectError:
        return False, "Cannot connect to SMTP server. Check your internet connection."
    except Exception as e:
        return False, f"Error: {str(e)}"


def send_congestion_alert(cfg: dict, location: str, vehicle_count: int,
                          frame_id: int, threshold: int, severity: str) -> tuple[bool, str]:
    subject = f"🚨 [{severity}] Traffic Congestion Alert — {location}"
    html    = _alert_html(location, vehicle_count, frame_id, threshold, severity)
    return send_email(
        cfg["sender_email"], cfg["sender_password"],
        cfg["recipient_email"], subject, html, cfg["provider"]
    )


def send_summary_email(cfg: dict, location: str, filename: str,
                       total_frames: int, max_vehicles: int,
                       congested_frames: int, congestion_rate: float,
                       threshold: int) -> tuple[bool, str]:
    level   = "HIGH" if congestion_rate > 50 else \
              "MODERATE" if congestion_rate > 20 else "LOW"
    subject = f"📊 Detection Report [{level}] — {location} ({congestion_rate:.1f}%)"
    html    = _summary_html(location, filename, total_frames, max_vehicles,
                            congested_frames, congestion_rate, threshold)
    return send_email(
        cfg["sender_email"], cfg["sender_password"],
        cfg["recipient_email"], subject, html, cfg["provider"]
    )
