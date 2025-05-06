import re

class AuthManager:
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    def check_password_strength(self, pwd: str) -> str | None:
        pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^\w\s]).{8,}$"
        if re.match(pattern, pwd):
            return None
        return "Password must be 8+ characters and include uppercase, lowercase, number, and symbol."

    def sign_in(self, email, password):
        try:
            result = self.supabase.auth.sign_in_with_password({"email": email, "password": password})
            return result.user
        except Exception as e:
            return f"Login failed: {e}"

    def sign_up(self, email, password):
        try:
            result = self.supabase.auth.sign_up({"email": email, "password": password})
            return result.user
        except Exception as e:
            return f"Signup failed: {e}"

    def change_password(self, new_password):
        try:
            self.supabase.auth.update_user({"password": new_password})
            return "Password updated."
        except Exception as e:
            return f"Password change failed: {e}"

    def sign_out(self):
        try:
            self.supabase.auth.sign_out()
            return "Signed out."
        except Exception as e:
            return f"Sign out failed: {e}"
