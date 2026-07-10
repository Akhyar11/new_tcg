const mysql = require('mysql2/promise');

async function createDb() {
  try {
    const connection = await mysql.createConnection({
      host: 'localhost',
      user: 'root',
      password: '123456',
      port: 3306
    });
    await connection.query('CREATE DATABASE IF NOT EXISTS pokemon_tcg;');
    console.log('Database pokemon_tcg created or already exists.');
    await connection.end();
  } catch (err) {
    console.error('Error creating database:', err.message);
  }
}
createDb();
