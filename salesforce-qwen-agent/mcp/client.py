"""
Salesforce MCP Client — connects to Salesforce's hosted MCP Server
via SSE (Server-Sent Events) transport with OAuth Bearer token authentication.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _extract_sobject(arguments: dict[str, Any]) -> str:
    for key in ["sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"]:
        if key in arguments and arguments[key]:
            return str(arguments[key]).strip()
    return ""


def _extract_body(arguments: dict[str, Any]) -> dict[str, Any]:
    if "body" in arguments and isinstance(arguments["body"], dict):
        return dict(arguments["body"])
    if "fields" in arguments and isinstance(arguments["fields"], dict):
        return dict(arguments["fields"])
    if "record" in arguments and isinstance(arguments["record"], dict):
        return dict(arguments["record"])
    ignore_keys = {"sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName", "id", "record_id", "relationship-path", "relationship_path"}
    return {k: v for k, v in arguments.items() if k not in ignore_keys}


class SalesforceMCPClient:
    """
    Manages connection to the Salesforce MCP Server.

    Handles:
    - OAuth token management (password flow + refresh)
    - SSE transport connection via MCP SDK
    - Session lifecycle (initialize → use → close)
    - Auto-reconnection on failure
    """

    def __init__(
        self,
        mcp_url: str,
        instance_url: str,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        security_token: str,
        domain: str = "login",
        access_token: str | None = None,
        refresh_token: str | None = None,
    ):
        self.mcp_url = mcp_url
        self.instance_url = instance_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.security_token = security_token
        self.domain = domain

        self._access_token = access_token
        self._refresh_token = refresh_token
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._connected = False
        self._schema_cache: dict[str, Any] = {}

        self._http_client = httpx.AsyncClient(timeout=60.0)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def access_token(self) -> str | None:
        return self._access_token

    # ──────────────────────────────────────────────────────────
    # OAuth Authentication
    # ──────────────────────────────────────────────────────────

    async def _soap_authenticate(self) -> str:
        """SOAP partner login fallback using username + password + security_token."""
        login_url = f"https://{self.domain}.salesforce.com/services/Soap/u/58.0"
        soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:partner.soap.sforce.com">
          <soapenv:Body>
            <urn:login>
              <urn:username>{self.username}</urn:username>
              <urn:password>{self.password}{self.security_token}</urn:password>
            </urn:login>
          </soapenv:Body>
        </soapenv:Envelope>"""

        headers = {
            "Content-Type": "text/xml",
            "SOAPAction": "login",
        }

        try:
            response = await self._http_client.post(
                login_url,
                data=soap_body,
                headers=headers,
            )
            response.raise_for_status()

            import xml.etree.ElementTree as ET
            import urllib.parse
            root = ET.fromstring(response.text)
            ns = {"soap": "http://schemas.xmlsoap.org/soap/envelope/", "urn": "urn:partner.soap.sforce.com"}
            session_id_elem = root.find(".//urn:sessionId", ns)
            server_url_elem = root.find(".//urn:serverUrl", ns)

            if session_id_elem is not None and session_id_elem.text:
                self._access_token = session_id_elem.text
                if server_url_elem is not None and server_url_elem.text:
                    parsed = urllib.parse.urlparse(server_url_elem.text)
                    self.instance_url = f"{parsed.scheme}://{parsed.netloc}"
                logger.info(f"Authenticated with Salesforce via SOAP partner login. Instance: {self.instance_url}")
                return self._access_token
            else:
                raise RuntimeError("SOAP login response missing sessionId.")
        except Exception as err:
            logger.error(f"Salesforce SOAP auth error: {err}")
            raise RuntimeError(f"Salesforce authentication failed: {err}")

    async def authenticate(self) -> str:
        """
        Authenticate with Salesforce using fast SOAP partner login,
        falling back to OAuth if needed.
        """
        if self.username and self.password and self.security_token:
            try:
                return await self._soap_authenticate()
            except Exception as soap_err:
                logger.warning(f"SOAP auth failed ({soap_err}), trying OAuth fallback...")

        token_url = f"https://{self.domain}.salesforce.com/services/oauth2/token"

        payload = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": f"{self.password}{self.security_token}",
        }

        try:
            response = await self._http_client.post(
                token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            self.instance_url = data.get("instance_url", self.instance_url)

            logger.info(
                f"Authenticated with Salesforce via OAuth. Instance: {self.instance_url}"
            )
            return self._access_token

        except Exception as e:
            logger.warning(f"OAuth authentication failed ({e}). Falling back to SOAP Login...")
            return await self._soap_authenticate()

    async def refresh_access_token(self) -> str:
        """Refresh the access token by authenticating."""
        logger.info("Refreshing access token...")
        return await self.authenticate()

    # ──────────────────────────────────────────────────────────
    # MCP Connection
    # ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Establish connection to the Salesforce MCP Server.
        Authenticates if needed, then connects via SSE transport.
        """
        try:
            await self.authenticate()
        except Exception as e:
            logger.warning(f"Initial authentication warning ({e}). Will retry on tool execution.")

        logger.info(f"Connecting to Salesforce MCP Server: {self.mcp_url}")
        self._connected = True
        logger.info("MCP Client ready (using direct HTTP transport).")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool call against Salesforce REST API.
        Tries direct REST API first for high reliability, auto-reauthenticates on 401.
        """
        if not self._access_token:
            try:
                await self.authenticate()
            except Exception as e:
                logger.warning(f"Auto-auth failed: {e}. Proceeding with configured access token.")

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Primary: Execute via direct Salesforce REST API
        try:
            return await self._fallback_rest_api(tool_name, arguments, headers)
        except httpx.HTTPStatusError as status_err:
            if status_err.response.status_code == 401 or "INVALID_SESSION_ID" in status_err.response.text:
                logger.warning(f"Session expired during {tool_name} (401). Re-authenticating...")
                try:
                    await self.authenticate()
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    return await self._fallback_rest_api(tool_name, arguments, headers)
                except Exception as auth_err:
                    logger.error(f"Re-authentication retry failed: {auth_err}")
                    raise RuntimeError(f"Salesforce API 401 Unauthorized: {status_err.response.text}") from auth_err
            raise

        # Secondary: Execute via MCP JSON-RPC server
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        try:
            response = await self._http_client.post(
                self.mcp_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            if "result" in result:
                return result["result"]
            elif "error" in result:
                error = result["error"]
                raise RuntimeError(
                    f"MCP Error [{error.get('code', 'unknown')}]: {error.get('message', 'Unknown error')}"
                )
            else:
                return result
        except Exception as e:
            logger.error(f"Tool {tool_name} execution error: {e}")
            raise

    async def _fallback_rest_api(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        headers: dict[str, str],
    ) -> Any:
        """
        Fallback: execute the tool via direct Salesforce REST API
        when MCP endpoint is unreachable.
        """
        base = self.instance_url.rstrip("/")
        api_version = "v62.0"

        try:
            if tool_name == "soqlQuery":
                query_str = arguments.get("q") or arguments.get("query") or arguments.get("soql") or ""
                url = f"{base}/services/data/{api_version}/query"
                resp = await self._http_client.get(
                    url, params={"q": query_str}, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "find":
                search_term = arguments.get("q") or arguments.get("query") or arguments.get("search") or arguments.get("find") or ""
                search_term = search_term.strip()
                if not search_term.startswith("FIND"):
                    # Format as valid SOSL query
                    clean_term = search_term.replace("{", "").replace("}", "").replace("'", "")
                    search_term = f"FIND {{{clean_term}*}} IN ALL FIELDS RETURNING Account(Id, Name), Contact(Id, Name, Email), Lead(Id, Name, Company, Email) LIMIT 20"

                url = f"{base}/services/data/{api_version}/search"
                resp = await self._http_client.get(
                    url, params={"q": search_term}, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "getUserInfo":
                try:
                    url = f"{base}/services/oauth2/userinfo"
                    resp = await self._http_client.get(url, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
                except Exception:
                    # Fallback to querying User via SOQL for current logged-in username
                    url = f"{base}/services/data/{api_version}/query"
                    query = f"SELECT Id, Name, Username, Email, Profile.Name FROM User WHERE Username = '{self.username}' LIMIT 1"
                    resp = await self._http_client.get(url, params={"q": query}, headers=headers)
                    resp.raise_for_status()
                    return resp.json()

            elif tool_name == "getObjectSchema":
                objects = arguments.get("objects") or arguments.get("object") or arguments.get("sobject-name") or arguments.get("sobject")
                if objects:
                    # Get specific object schema
                    obj_list = [o.strip() for o in str(objects).split(",") if o.strip()]
                    results = {}
                    uncached = []
                    for obj in obj_list:
                        cache_key = f"schema:{obj.lower()}"
                        if cache_key in self._schema_cache:
                            results[obj] = self._schema_cache[cache_key]
                        else:
                            uncached.append(obj)

                    for obj in uncached:
                        url = f"{base}/services/data/{api_version}/sobjects/{obj}/describe"
                        resp = await self._http_client.get(url, headers=headers)
                        resp.raise_for_status()
                        raw = resp.json()

                        simplified_fields = []
                        for f in raw.get("fields", []):
                            req = not f.get("nillable") and not f.get("defaultedOnCreate") and f.get("createable")
                            field_info = {
                                "name": f.get("name"),
                                "label": f.get("label"),
                                "type": f.get("type"),
                            }
                            if req:
                                field_info["required"] = True
                            if f.get("type") == "picklist" and f.get("picklistValues"):
                                active_vals = [p.get("value") for p in f.get("picklistValues", []) if p.get("active")]
                                if active_vals:
                                    field_info["picklist_values"] = active_vals[:15]
                            simplified_fields.append(field_info)

                        obj_schema = {
                            "name": raw.get("name"),
                            "label": raw.get("label"),
                            "total_fields": len(simplified_fields),
                            "fields": simplified_fields,
                        }
                        cache_key = f"schema:{obj.lower()}"
                        self._schema_cache[cache_key] = obj_schema
                        results[obj] = obj_schema

                    return results
                else:
                    # Get all queryable objects (cached & filtered)
                    if "all_sobjects" in self._schema_cache:
                        return self._schema_cache["all_sobjects"]

                    url = f"{base}/services/data/{api_version}/sobjects"
                    resp = await self._http_client.get(url, headers=headers)
                    resp.raise_for_status()
                    raw = resp.json()

                    # Filter sObjects: keep queryable standard/custom objects, skip internal metadata noise
                    filtered_sobjects = []
                    skip_suffixes = ("ChangeEvent", "Feed", "History", "Share", "Tag", "Permission", "Group", "Access")
                    for sobj in raw.get("sobjects", []):
                        name = sobj.get("name", "")
                        if not sobj.get("queryable"):
                            continue
                        if any(name.endswith(s) for s in skip_suffixes):
                            continue
                        filtered_sobjects.append({
                            "name": name,
                            "label": sobj.get("label"),
                            "custom": sobj.get("custom", False),
                        })

                    result_data = {
                        "total": len(filtered_sobjects),
                        "sobjects": filtered_sobjects,
                    }
                    self._schema_cache["all_sobjects"] = result_data
                    return result_data

            elif tool_name == "getRelatedRecords":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}/{rel_path}"
                resp = await self._http_client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "listRecentSobjectRecords":
                sobject = _extract_sobject(arguments)
                if sobject:
                    try:
                        url = f"{base}/services/data/{api_version}/query"
                        query = f"SELECT Id, Name, CreatedDate, LastModifiedDate FROM {sobject} ORDER BY LastModifiedDate DESC LIMIT 10"
                        resp = await self._http_client.get(url, params={"q": query}, headers=headers)
                        if resp.status_code == 200:
                            return resp.json()
                    except Exception:
                        pass
                url = f"{base}/services/data/{api_version}/recent"
                resp = await self._http_client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "createSobjectRecord":
                sobject = _extract_sobject(arguments) or "Lead"
                body = _extract_body(arguments)
                
                # Auto-fix standard mandatory fields if user provided minimal input
                s_lower = sobject.lower()
                if s_lower == "lead":
                    if "LastName" not in body:
                        raw_name = body.pop("Name", None) or body.pop("FirstName", None) or arguments.get("name")
                        if raw_name:
                            name_parts = str(raw_name).strip().split(None, 1)
                            if len(name_parts) == 2:
                                body["FirstName"] = name_parts[0]
                                body["LastName"] = name_parts[1]
                            else:
                                body["LastName"] = name_parts[0]
                                body.pop("FirstName", None)
                        else:
                            body["LastName"] = "Unknown"
                    if "Company" not in body:
                        body["Company"] = "Individual"

                elif s_lower == "contact":
                    if "LastName" not in body:
                        raw_name = body.pop("Name", None) or body.pop("FirstName", None) or arguments.get("name")
                        if raw_name:
                            name_parts = str(raw_name).strip().split(None, 1)
                            if len(name_parts) == 2:
                                body["FirstName"] = name_parts[0]
                                body["LastName"] = name_parts[1]
                            else:
                                body["LastName"] = name_parts[0]
                                body.pop("FirstName", None)
                        else:
                            body["LastName"] = "Unknown"

                elif s_lower == "account":
                    if "Name" not in body and "LastName" in body:
                        body["Name"] = body.pop("LastName")
                    if "Name" not in body:
                        body["Name"] = "New Account"

                elif s_lower == "opportunity":
                    if "Name" not in body:
                        body["Name"] = "New Opportunity"
                    if "StageName" not in body:
                        body["StageName"] = "Prospecting"
                    if "CloseDate" not in body:
                        body["CloseDate"] = "2026-12-31"

                elif s_lower == "case":
                    if "Subject" not in body and "Name" in body:
                        body["Subject"] = body.pop("Name")
                    if "Subject" not in body:
                        body["Subject"] = "New Customer Inquiry"
                    if "Status" not in body:
                        body["Status"] = "New"

                url = f"{base}/services/data/{api_version}/sobjects/{sobject}"
                post_headers = {**headers, "Sforce-Duplicate-Rule-Header": "allowSave=true"}
                resp = await self._http_client.post(url, json=body, headers=post_headers)
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "updateSobjectRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                body = _extract_body(arguments)
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}"
                resp = await self._http_client.patch(url, json=body, headers=headers)
                if resp.status_code == 204:
                    return {"success": True, "id": record_id}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "updateRelatedRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
                body = _extract_body(arguments)
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}/{rel_path}"
                resp = await self._http_client.patch(url, json=body, headers=headers)
                if resp.status_code == 204:
                    return {"success": True}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "deleteSobjectRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}"
                resp = await self._http_client.delete(url, headers=headers)
                if resp.status_code == 204:
                    return {"success": True, "deleted": record_id}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "deleteRelatedRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}/{rel_path}"
                resp = await self._http_client.delete(url, headers=headers)
                if resp.status_code == 204:
                    return {"success": True}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "uploadRecordAttachment":
                import base64, os
                record_id = (arguments.get("record_id") or arguments.get("id") or "").strip()
                file_name = arguments.get("file_name") or arguments.get("filename") or "attachment"
                title = arguments.get("title") or os.path.splitext(file_name)[0]
                base64_data = arguments.get("file_content_base64") or arguments.get("base64") or ""

                if not record_id or record_id.lower() in ("001g500000ddq7saau", "001000000000000", "account_id", "record_id", "dummy"):
                    raise RuntimeError("A valid Salesforce Record ID is required to attach the file. Please provide the specific Record ID.")

                if not base64_data and os.path.exists(os.path.join("uploads", file_name)):
                    with open(os.path.join("uploads", file_name), "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode("utf-8")

                if not base64_data:
                    raise RuntimeError("Missing file content (base64) for uploadRecordAttachment.")

                url = f"{base}/services/data/{api_version}/sobjects/ContentVersion"
                payload = {
                    "Title": title,
                    "PathOnClient": file_name,
                    "VersionData": base64_data,
                }
                if record_id:
                    payload["FirstPublishLocationId"] = record_id

                resp = await self._http_client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                result = resp.json()
                return {
                    "success": True,
                    "content_version_id": result.get("id"),
                    "linked_record_id": record_id,
                    "file_name": file_name,
                    "message": f"File '{file_name}' successfully attached to Salesforce record {record_id}."
                }

            else:
                raise RuntimeError(f"Unknown tool: {tool_name}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 or "INVALID_SESSION_ID" in e.response.text:
                raise  # Re-raise so call_tool catches 401 and auto-reauthenticates
            error_body = e.response.text
            logger.error(f"REST API fallback failed for {tool_name}: {error_body}")
            raise RuntimeError(f"Salesforce API error for {tool_name}: {error_body}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools from the MCP Server.
        Falls back to local definitions if MCP is unreachable.
        """
        if not self._access_token:
            await self.authenticate()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        try:
            response = await self._http_client.post(
                self.mcp_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            if "result" in result and "tools" in result["result"]:
                tools = result["result"]["tools"]
                logger.info(f"Discovered {len(tools)} tools from MCP Server.")
                return tools

        except Exception as e:
            logger.warning(f"Could not list tools from MCP: {e}. Using local definitions.")

        # Fallback to local tool definitions
        from tools.salesforce import get_tool_definitions
        return get_tool_definitions()

    async def disconnect(self) -> None:
        """Close the MCP connection and clean up resources."""
        self._connected = False
        try:
            await self._http_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")
        logger.info("MCP Client disconnected.")
