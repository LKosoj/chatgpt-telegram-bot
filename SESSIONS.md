## Guidelines for managing sessions in Telegram bot

## Introduction
Session management functionality allows you to create, switch and manage multiple contexts of communication with the bot, increasing the efficiency and flexibility of your interactions.

## Main features
- Create up to MAX_SESSIONS of active sessions (set in the MAX_SESSIONS environment variable)
- Automatic generation of unique session names
- Switch between sessions
- View session history

## Session management commands

### Create a new session
- Use the `/new_session` command
- The first query in the new session will automatically determine its name.

### Switching between sessions
- Use the `/switch_session` command
- You will be shown a list of available sessions
- Select the session to activate

### View the list of sessions
- The `/list_sessions` command will show all your sessions
- The information displayed is:
  - Session name
  - Date created
  - Number of messages
  - Activity status

### Deleting a session
- Command `/delete_session`
- When deleting an active session, the previous session is automatically activated.

### Features of sessions
- Maximum MAX_SESSIONS of active sessions (set in the MAX_SESSIONS environment variable)
- When MAX_SESSIONS+1th session is created, the oldest session will be automatically deleted
- Each session has its own independent context
- Session settings are inherited from the previous session

### Usage Examples

### Scenario 1: Different tasks
1. Programming session for technical issues
2. “Travel” session for trip planning
3. “Cooking” session for recipes

### Scenario 2: Separate Contexts
- Separate session for each project
- Keeping contexts without mixing tasks

### Tips
- Give meaningful names to sessions
- Regularly clean up irrelevant sessions
- Use sessions to organize information

## Limitations
- Maximum MAX_SESSIONS of active sessions
- Session history is kept for a limited time
- Unable to restore a deleted session 