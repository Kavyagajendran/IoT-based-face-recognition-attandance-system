import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    registered_date TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    check_in TEXT NOT NULL,
                    check_out TEXT,
                    FOREIGN KEY (employee_id) REFERENCES employees (id)
                )
            ''')
            conn.commit()

    def employee_exists(self, name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM employees WHERE name = ?', (name,))
            return cursor.fetchone() is not None

    def add_employee(self, name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            registered_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('INSERT INTO employees (name, registered_date) VALUES (?, ?)', 
                         (name, registered_date))
            conn.commit()

    def get_employee_id(self, name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM employees WHERE name = ?', (name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def mark_attendance(self, name, check_in, check_out):
        employee_id = self.get_employee_id(name)
        if not employee_id:
            return False
        
        attendance_date = check_in.split()[0]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO attendance (employee_id, date, check_in, check_out)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, attendance_date, check_in, check_out))
            conn.commit()
        return True

    def update_checkout(self, name, date_str, check_out):
        employee_id = self.get_employee_id(name)
        if not employee_id:
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE attendance 
                SET check_out = ?
                WHERE employee_id = ? AND date = ? AND check_out IS NULL
            ''', (check_out, employee_id, date_str))
            conn.commit()
        return True

    def has_checked_in(self, name, date_str):
        #print(f"name={name}")
        employee_id = self.get_employee_id(name)
        if not employee_id:
            return False
        #print(f"employee_id={employee_id}, date_str={date_str}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE employee_id = ? AND date = ?
            ''', (employee_id, date_str))
            return cursor.fetchone() is not None

    def has_checked_out(self, name, date_str):
        employee_id = self.get_employee_id(name)
        if not employee_id:
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE employee_id = ? AND date = ? AND check_out IS NOT NULL
            ''', (employee_id, date_str))
            return cursor.fetchone() is not None

    def get_todays_attendance(self, date_str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column name access
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.name, a.check_in, a.check_out 
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.date = ?
                ORDER BY a.check_in
            ''', (date_str,))
            return [dict(row) for row in cursor.fetchall()]
            #return cursor.fetchall()

    def get_attendance_between_dates(self, from_date, to_date):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column name access
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.id, e.name, a.date, a.check_in, a.check_out 
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.date BETWEEN ? AND ?
                ORDER BY a.date, a.check_in
            ''', (from_date, to_date))
            return [dict(row) for row in cursor.fetchall()]
            #return cursor.fetchall()

    def delete_attendance_record(self, record_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance WHERE id = ?', (record_id,))
            conn.commit()

    def get_all_employees(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, registered_date FROM employees ORDER BY name')
            return cursor.fetchall()

    def delete_employee(self, employee_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance WHERE employee_id = ?', (employee_id,))
            cursor.execute('DELETE FROM employees WHERE id = ?', (employee_id,))
            conn.commit()