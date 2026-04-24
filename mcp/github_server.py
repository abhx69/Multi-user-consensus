"""
GitHub MCP Server.

Provides tools for GitHub operations:
- Create issues
- List repositories
- Search issues
- Get repository info
"""

import logging
from typing import Any

from gaprio.config import settings
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


class GitHubMCPServer(BaseMCPServer):
    """
    MCP server for GitHub operations.
    
    Provides tools for the agent to interact with GitHub:
    - Creating issues from discussions
    - Managing repositories
    - Searching for related issues
    
    Usage:
        server = GitHubMCPServer()
        result = await server.execute(
            "github_create_issue",
            repo="my-repo",
            title="Bug: ...",
            body="..."
        )
    """
    
    def __init__(self):
        """Initialize the GitHub MCP server."""
        self._github = None
        super().__init__("github")
    
    def _get_github(self):
        """Lazy initialization of GitHub client."""
        if self._github is None:
            from github import Github
            
            token = settings.github_token
            if not token:
                raise RuntimeError("GitHub token not configured")
            
            self._github = Github(token)
        
        return self._github
    
    def _register_tools(self) -> None:
        """Register all GitHub tools."""
        
        # Create issue
        self.add_tool(MCPTool(
            name="github_create_issue",
            description=(
                "Create a new issue in a GitHub repository. "
                "Use for tracking bugs, feature requests, or tasks from Slack discussions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository name (e.g., 'my-repo' or 'owner/repo')",
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title",
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body/description in markdown",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to add (optional)",
                    },
                    "assignees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "GitHub usernames to assign (optional)",
                    },
                },
                "required": ["repo", "title", "body"],
            },
            handler=self._create_issue,
        ))
        
        # List repositories
        self.add_tool(MCPTool(
            name="github_list_repos",
            description="List repositories accessible to the bot.",
            parameters={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Repository type: all, owner, public, private, member",
                        "default": "all",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum repositories to return",
                        "default": 20,
                    },
                },
            },
            handler=self._list_repos,
        ))
        
        # Get repository info
        self.add_tool(MCPTool(
            name="github_get_repo",
            description="Get information about a specific repository.",
            parameters={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository name (e.g., 'my-repo' or 'owner/repo')",
                    },
                },
                "required": ["repo"],
            },
            handler=self._get_repo,
        ))
        
        # Search issues
        self.add_tool(MCPTool(
            name="github_search_issues",
            description="Search for issues across repositories.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Limit to specific repo (optional)",
                    },
                    "state": {
                        "type": "string",
                        "description": "Issue state: open, closed, all",
                        "default": "open",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            handler=self._search_issues,
        ))
        
        # List open issues
        self.add_tool(MCPTool(
            name="github_list_issues",
            description="List issues in a repository.",
            parameters={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                    "state": {
                        "type": "string",
                        "description": "Issue state: open, closed, all",
                        "default": "open",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20,
                    },
                },
                "required": ["repo"],
            },
            handler=self._list_issues,
        ))
    
    def _get_full_repo_name(self, repo: str) -> str:
        """Get full repo name (owner/repo) from short name."""
        if "/" in repo:
            return repo
        
        owner = settings.github_default_owner
        if not owner:
            raise ValueError("No owner specified and GITHUB_DEFAULT_OWNER not set")
        
        return f"{owner}/{repo}"
    
    async def _create_issue(
        self,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> ToolResult:
        """
        Create a GitHub issue.
        
        Args:
            repo: Repository name
            title: Issue title
            body: Issue description
            labels: Optional labels
            assignees: Optional assignees
            
        Returns:
            ToolResult with issue URL
        """
        try:
            gh = self._get_github()
            full_repo = self._get_full_repo_name(repo)
            
            repository = gh.get_repo(full_repo)
            
            # Create the issue
            issue = repository.create_issue(
                title=title,
                body=body,
                labels=labels or [],
                assignees=assignees or [],
            )
            
            logger.info(f"Created issue #{issue.number} in {full_repo}")
            
            return ToolResult(
                success=True,
                data={
                    "number": issue.number,
                    "url": issue.html_url,
                    "title": issue.title,
                    "repo": full_repo,
                    "message": f"Created issue #{issue.number}: {issue.title}",
                },
            )
            
        except Exception as e:
            logger.error(f"Error creating issue: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _list_repos(
        self,
        type: str = "all",
        limit: int = 20,
    ) -> ToolResult:
        """List accessible repositories."""
        try:
            gh = self._get_github()
            user = gh.get_user()
            
            repos = []
            for i, repo in enumerate(user.get_repos(type=type)):
                if i >= limit:
                    break
                repos.append({
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "private": repo.private,
                    "url": repo.html_url,
                })
            
            return ToolResult(
                success=True,
                data={
                    "repos": repos,
                    "count": len(repos),
                },
            )
            
        except Exception as e:
            logger.error(f"Error listing repos: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_repo(self, repo: str) -> ToolResult:
        """Get repository information."""
        try:
            gh = self._get_github()
            full_repo = self._get_full_repo_name(repo)
            
            repository = gh.get_repo(full_repo)
            
            return ToolResult(
                success=True,
                data={
                    "name": repository.name,
                    "full_name": repository.full_name,
                    "description": repository.description,
                    "private": repository.private,
                    "url": repository.html_url,
                    "default_branch": repository.default_branch,
                    "open_issues": repository.open_issues_count,
                    "stars": repository.stargazers_count,
                    "language": repository.language,
                },
            )
            
        except Exception as e:
            logger.error(f"Error getting repo: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_issues(
        self,
        query: str,
        repo: str | None = None,
        state: str = "open",
        limit: int = 10,
    ) -> ToolResult:
        """Search for issues."""
        try:
            gh = self._get_github()
            
            # Build search query
            search_query = query
            if repo:
                full_repo = self._get_full_repo_name(repo)
                search_query += f" repo:{full_repo}"
            if state != "all":
                search_query += f" state:{state}"
            
            issues = []
            for i, issue in enumerate(gh.search_issues(search_query)):
                if i >= limit:
                    break
                issues.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "url": issue.html_url,
                    "repo": issue.repository.full_name,
                    "created_at": issue.created_at.isoformat(),
                })
            
            return ToolResult(
                success=True,
                data={
                    "issues": issues,
                    "count": len(issues),
                    "query": search_query,
                },
            )
            
        except Exception as e:
            logger.error(f"Error searching issues: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _list_issues(
        self,
        repo: str,
        state: str = "open",
        limit: int = 20,
    ) -> ToolResult:
        """List issues in a repository."""
        try:
            gh = self._get_github()
            full_repo = self._get_full_repo_name(repo)
            
            repository = gh.get_repo(full_repo)
            
            issues = []
            for i, issue in enumerate(repository.get_issues(state=state)):
                if i >= limit:
                    break
                # Skip pull requests (they show up as issues too)
                if issue.pull_request:
                    continue
                issues.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "url": issue.html_url,
                    "labels": [l.name for l in issue.labels],
                    "created_at": issue.created_at.isoformat(),
                })
            
            return ToolResult(
                success=True,
                data={
                    "issues": issues,
                    "count": len(issues),
                    "repo": full_repo,
                },
            )
            
        except Exception as e:
            logger.error(f"Error listing issues: {e}")
            return ToolResult(success=False, error=str(e))
