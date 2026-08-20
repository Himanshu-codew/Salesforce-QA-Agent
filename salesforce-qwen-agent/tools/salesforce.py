"""
Salesforce tool definitions for the 11 MCP tools.
Used as a fallback registry when MCP server tool discovery is unavailable,
and as the canonical source for OpenAI function calling format conversion.
"""

from typing import Any

# ──────────────────────────────────────────────────────────────
# Tool Categories
# ──────────────────────────────────────────────────────────────
READ_ONLY_TOOLS = {
    "getRelatedRecords",
    "listRecentSobjectRecords",
    "soqlQuery",
    "find",
    "getUserInfo",
    "getObjectSchema",
}

MUTATING_TOOLS = {
    "createSobjectRecord",
    "updateSobjectRecord",
    "updateRelatedRecord",
    "uploadRecordAttachment",
}

DESTRUCTIVE_TOOLS = {
    "deleteSobjectRecord",
    "deleteRelatedRecord",
}

# ──────────────────────────────────────────────────────────────
# Complete Tool Definitions (OpenAI Function Calling Format)
# ──────────────────────────────────────────────────────────────
SALESFORCE_TOOLS: list[dict[str, Any]] = [
    # 1. Get Related Records
    {
        "type": "function",
        "function": {
            "name": "getRelatedRecords",
            "description": "Retrieves child records related to a parent record by ID and relationship name (e.g. sobject-name='Account', relationship-path='Contacts').",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of parent object (e.g., 'Account').",
                    },
                    "id": {
                        "type": "string",
                        "description": "ID of parent record.",
                    },
                    "relationship-path": {
                        "type": "string",
                        "description": "Relationship path (e.g., 'Contacts').",
                    },
                },
                "required": ["sobject-name", "id", "relationship-path"],
            },
        },
    },
    # 2. List Recently Viewed Records
    {
        "type": "function",
        "function": {
            "name": "listRecentSobjectRecords",
            "description": "Returns recently viewed or modified records of a given object type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of sObject (e.g., 'Account', 'Case', 'Opportunity').",
                    },
                },
                "required": ["sobject-name"],
            },
        },
    },
    # 3. SOQL Query
    {
        "type": "function",
        "function": {
            "name": "soqlQuery",
            "description": "Executes a SOQL query to read Salesforce data. Example: SELECT Id, Name, Account.Name FROM Opportunity WHERE Amount > 1000 ORDER BY Amount DESC LIMIT 10",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "SOQL query string.",
                    },
                },
                "required": ["q"],
            },
        },
    },
    # 4. Search Across Objects (SOSL)
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Executes text search across multiple objects using SOSL. Example: FIND {Acme} IN ALL FIELDS RETURNING Account(Id, Name), Contact(Id, Name)",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "SOSL search query string.",
                    },
                },
                "required": ["q"],
            },
        },
    },
    # 5. Get Current User Info
    {
        "type": "function",
        "function": {
            "name": "getUserInfo",
            "description": "Returns current logged-in Salesforce user profile, role, email, and identity details.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # 6. Get Object Schema
    {
        "type": "function",
        "function": {
            "name": "getObjectSchema",
            "description": "Returns object schema details (fields, types, required flags, picklist values). Only call when field API names are unknown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "string",
                        "description": "Comma-separated object API names (e.g., 'Account', 'Lead').",
                    },
                },
                "required": [],
            },
        },
    },
    # 7. Create Record
    {
        "type": "function",
        "function": {
            "name": "createSobjectRecord",
            "description": "Creates a new Salesforce record. Requires: sobject-name and body (field-value map).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of the object (e.g., 'Account', 'Contact').",
                    },
                    "body": {
                        "type": "object",
                        "description": "Field-value pairs for the new record.",
                    },
                },
                "required": ["sobject-name", "body"],
            },
        },
    },
    # 8. Update Record
    {
        "type": "function",
        "function": {
            "name": "updateSobjectRecord",
            "description": "Updates an existing record by ID with provided field-value pairs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of the object.",
                    },
                    "id": {
                        "type": "string",
                        "description": "ID of record to update.",
                    },
                    "body": {
                        "type": "object",
                        "description": "Field-value pairs to update.",
                    },
                },
                "required": ["sobject-name", "id", "body"],
            },
        },
    },
    # 9. Update Related Record
    {
        "type": "function",
        "function": {
            "name": "updateRelatedRecord",
            "description": "Updates a child record by navigating from parent record ID through a relationship path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of parent object.",
                    },
                    "id": {
                        "type": "string",
                        "description": "ID of parent record.",
                    },
                    "relationship-path": {
                        "type": "string",
                        "description": "Relationship path.",
                    },
                    "body": {
                        "type": "object",
                        "description": "Field-value pairs to update.",
                    },
                },
                "required": ["sobject-name", "id", "relationship-path", "body"],
            },
        },
    },
    # 10. Delete Record
    {
        "type": "function",
        "function": {
            "name": "deleteSobjectRecord",
            "description": "Permanently deletes a record by sObject name and ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of the object.",
                    },
                    "id": {
                        "type": "string",
                        "description": "ID of record to delete.",
                    },
                },
                "required": ["sobject-name", "id"],
            },
        },
    },
    # 11. Delete Related Record
    {
        "type": "function",
        "function": {
            "name": "deleteRelatedRecord",
            "description": "Deletes a child record related to a parent record by relationship path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sobject-name": {
                        "type": "string",
                        "description": "API name of parent object.",
                    },
                    "id": {
                        "type": "string",
                        "description": "ID of parent record.",
                    },
                    "relationship-path": {
                        "type": "string",
                        "description": "Relationship path to child record.",
                    },
                },
                "required": ["sobject-name", "id", "relationship-path"],
            },
        },
    },
    # 12. Upload Record Attachment / File
    {
        "type": "function",
        "function": {
            "name": "uploadRecordAttachment",
            "description": "Uploads and attaches a file/document to a Salesforce record (Account, Contact, Lead, Opportunity, Case, etc.) via ContentVersion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "The valid 15 or 18 character Salesforce Record ID to attach the file to (e.g., specific Account ID, Opportunity ID, or Lead ID) explicitly provided by the user. Do NOT use fake, dummy, or placeholder IDs.",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file including extension (e.g. 'Proposal.pdf', 'Agreement.docx', 'leads.csv').",
                    },
                    "file_content_base64": {
                        "type": "string",
                        "description": "Base64 encoded string of the file content.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the attached document.",
                    },
                },
                "required": ["record_id", "file_name"],
            },
        },
    },
]


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return all Salesforce tool definitions in OpenAI function calling format."""
    return SALESFORCE_TOOLS


def is_destructive(tool_name: str) -> bool:
    """Check if a tool is destructive (delete operations)."""
    return tool_name in DESTRUCTIVE_TOOLS


def is_mutating(tool_name: str) -> bool:
    """Check if a tool modifies data (create/update/delete)."""
    return tool_name in MUTATING_TOOLS or tool_name in DESTRUCTIVE_TOOLS


def is_read_only(tool_name: str) -> bool:
    """Check if a tool is read-only."""
    return tool_name in READ_ONLY_TOOLS
